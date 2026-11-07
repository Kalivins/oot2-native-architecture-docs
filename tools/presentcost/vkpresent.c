// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkpresent: the windowed present spike (architecture D22, D23, D26 and Q4).
// A bare Win32 window, a producer of synthetic frames paced on a virtual 60 Hz
// clock, and a present thread that shows each new frame through a Vulkan
// swapchain, with three instruments that say when a frame reached the screen:
//
//   present timing   VK_EXT_present_timing results per present id: the
//                    queue-operations-end and first-pixel-out stages, in the
//                    swapchain's time domain, calibrated against the
//                    performance counter with vkGetCalibratedTimestampsKHR
//   present wait     a wait thread calls vkWaitForPresent2KHR for every
//                    presented id (VK_KHR_present_id2) and notes the counter
//                    when it returns
//   vblanks          a thread notes the counter at every vertical blank of
//                    the window's display (D3DKMTWaitForVerticalBlankEvent), an
//                    independent grid that needs no swapchain
//
//   vkpresent run --window composed|borderless --mode fifo|relaxed|mailbox|latest|immediate
//                 --out PREFIX [--frames N] [--timing on|off] [--relative NS] [--nearest]
//                 [--skip K] [--images N] [--slice-us N] [--wait-lock on|off] [--occlude]
//
// composed is a decorated 1280x720 window that the desktop composes;
// borderless is a popup at the size of its monitor (the independent-flip
// candidate, no mode change). Both stay topmost for the run. --occlude puts a
// small topmost window over the swapchain window's top-left corner, the way a
// notification or an overlay would, so the desktop has to compose the frame. The producer
// thread waits for tick n of the virtual clock (tick n is due at the start
// time plus n/60 s, on the performance counter) and hands frame n to the
// present thread through a one-slot mailbox: a frame the present thread has
// not taken when the next arrives is dropped by the host and counted. --skip K
// makes the producer skip every Kth tick (the counter-run). The present thread
// acquires, clears the image to a dark grey with a lighter bar that moves one
// step a tick (no flashing), submits and presents with present id n + 1 and,
// with --timing on (the default), a VkPresentTimingsInfoEXT asking for both
// stages; --relative NS adds VK_PRESENT_TIMING_INFO_PRESENT_AT_RELATIVE_TIME
// with targetTime NS (the minimum time the previous image stays visible), and
// --nearest the nearest-refresh flag. vkWaitForPresent2KHR and
// vkQueuePresentKHR both need the swapchain externally synchronised, so the
// wait thread waits in slices of --slice-us (default 1000) under a lock the
// present thread takes for vkQueuePresentKHR; the time the present thread
// spends on that lock is recorded. Each slice that times out costs up to a
// slice of delay in noticing the present, so --wait-lock off measures the
// wait's own wake-up instead: one unbounded wait per id, without the lock,
// which breaks the external-synchronisation rule and is for measurement only.
//
// Out: PREFIX.frames.tsv (one row per tick), PREFIX.vblanks.tsv (one counter
// value per vertical blank) and key=value lines on stdout: the environment
// (implicit-layer switch, every Vulkan layer DLL loaded into the process), the
// display mode, the desktop compositor's timing, the surface's present-timing
// capabilities, the swapchain, its timing properties (refreshDuration,
// refreshInterval: UINT64_MAX means variable refresh) as they change, its time
// domains, the calibration pairs, the NVIDIA driver's variable-refresh state of
// the primary display before the run and half way through it (NVAPI
// NvAPI_Disp_GetVRRInfo, read only; absent on other vendors) and a result line. The counter frequency and
// the TSC rate (the unit of the present thread's cycle counts) are printed so
// the caller can convert. Third-party implicit layers are the caller's to
// disable (VK_LOADER_LAYERS_DISABLE=~implicit~). Windows only; no game data.
//
// A watchdog ends a run that stops moving: when no present call returns and no
// present wait ends for 5 s, it prints a `stall` line (the last tick produced,
// presented and waited for, and where the present thread is), writes the rows
// it has and exits 4 without waiting for the threads, which are then blocked
// inside the driver.
//
// Exit codes: 0 done, 1 bad arguments or a run that did not finish (the window
// was lost, a present failed), 2 Vulkan or the display lacks what the run
// needs, 3 a Vulkan call failed, 4 the run stalled.

#define WIN32_LEAN_AND_MEAN
#define VK_NO_PROTOTYPES
#include "volk.h"

#include <windows.h>
#include <dwmapi.h>
#include <x86intrin.h>

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_IMAGES 8
#define IN_FLIGHT 2
#define TIMING_BATCH 64
#define MAX_DOMAINS 8
#define TICK_HZ 60

enum { ST_NONE = 0, ST_SKIPPED, ST_PRODUCED, ST_HOST_DROPPED, ST_PRESENTED };

typedef struct {
    int64_t due, produced, wake, acquired, present_call, present_ret, wait_ret;
    int64_t lock_wait;              // counter counts the present thread waited for the swapchain lock
    uint64_t pt_qoe, pt_fpo;        // present timing, raw, in the result's time domain
    uint64_t cycles;                // present thread cycles from wake to the return of vkQueuePresentKHR
    int32_t acq_res, pres_res, wait_res;
    uint32_t image;
    uint8_t state, pt_have, pt_complete, pt_domain;
} Rec;

static struct {
    int borderless;
    VkPresentModeKHR mode;
    const char *mode_name, *out;
    int timing, nearest;
    uint64_t relative_ns;
    uint32_t frames, skip, images, slice_us;
    int wait_lock, occlude;
} opt = {0, VK_PRESENT_MODE_FIFO_KHR, "fifo", NULL, 1, 0, 0, 10000, 0, 3, 1000, 1, 0};

static VkInstance inst;
static VkPhysicalDevice gpu;
static VkDevice dev;
static VkQueue queue;
static uint32_t family = UINT32_MAX;
static VkSurfaceKHR surface;
static VkSwapchainKHR swapchain;
static uint32_t nimages;
static VkImage images[MAX_IMAGES];
static VkImageView views[MAX_IMAGES];
static VkSemaphore done_sem[MAX_IMAGES];
static VkExtent2D extent;
static PFN_vkCreateWin32SurfaceKHR create_win32_surface;

static HWND hwnd;
static RECT mon_rect;
static WCHAR mon_device[32];
static int64_t qpf, t0;
static Rec *rec;
static volatile LONG mailbox;          // 0 or tick + 1
static volatile LONG producer_done, present_done, window_lost;
static HANDLE ev_frame, ev_pushed;
static SRWLOCK sc_lock = SRWLOCK_INIT;
static volatile LONG present_wants_lock;   // the wait thread steps aside while it is set (SRW locks are not fair)
static uint64_t *pushed;                // present ids in present order
static volatile LONG npushed;
static int64_t *vblanks;
static volatile LONG nvblanks;
static uint32_t max_vblanks;
static uint64_t domain_id;              // the time domain id passed with every present
static VkTimeDomainKHR domain_kind;
static struct { VkTimeDomainKHR kind; uint64_t id; } seen_domains[MAX_DOMAINS];
static uint32_t nseen;
static uint64_t last_props_counter = UINT64_MAX;
static int64_t present_thread_kernel100ns, present_thread_user100ns;
static uint32_t presented_frames;
static int failed_code;
static volatile LONG present_step;       // where the present thread is: 1 fence, 2 acquire, 3 submit, 4 lock, 5 present
static volatile LONG last_presented = -1, last_waited = -1, last_produced = -1;

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkpresent: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stdout);
    ExitProcess((UINT)code);
}

#define VKC(call)                                                                                  \
    do {                                                                                           \
        VkResult r_ = (call);                                                                      \
        if (r_ != VK_SUCCESS) die(3, "%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); \
    } while (0)

static int64_t qpc(void) {
    LARGE_INTEGER v;
    QueryPerformanceCounter(&v);
    return v.QuadPart;
}

static void usage(void) {
    fprintf(stderr,
            "usage: vkpresent run --window composed|borderless --mode fifo|relaxed|mailbox|latest|immediate\n"
            "                     --out PREFIX [--frames N] [--timing on|off] [--relative NS] [--nearest]\n"
            "                     [--skip K] [--images N] [--slice-us N] [--wait-lock on|off] [--occlude]\n");
    ExitProcess(1);
}

static uint64_t parse_u64(const char *s) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!*s || *end) usage();
    return v;
}

static void parse_args(int argc, char **argv) {
    if (argc < 2 || strcmp(argv[1], "run")) usage();
    int have_window = 0, have_mode = 0;
    for (int i = 2; i < argc; i++) {
        const char *a = argv[i];
        const char *v = i + 1 < argc ? argv[i + 1] : NULL;
        if (!strcmp(a, "--nearest")) { opt.nearest = 1; continue; }
        if (!strcmp(a, "--occlude")) { opt.occlude = 1; continue; }
        if (!v) usage();
        i++;
        if (!strcmp(a, "--window")) {
            if (!strcmp(v, "composed")) opt.borderless = 0;
            else if (!strcmp(v, "borderless")) opt.borderless = 1;
            else usage();
            have_window = 1;
        } else if (!strcmp(a, "--mode")) {
            static const struct { const char *name; VkPresentModeKHR mode; } modes[] = {
                {"fifo", VK_PRESENT_MODE_FIFO_KHR}, {"relaxed", VK_PRESENT_MODE_FIFO_RELAXED_KHR},
                {"mailbox", VK_PRESENT_MODE_MAILBOX_KHR}, {"latest", VK_PRESENT_MODE_FIFO_LATEST_READY_KHR},
                {"immediate", VK_PRESENT_MODE_IMMEDIATE_KHR}};
            int found = 0;
            for (size_t m = 0; m < sizeof modes / sizeof modes[0]; m++)
                if (!strcmp(v, modes[m].name)) { opt.mode = modes[m].mode; opt.mode_name = modes[m].name; found = 1; }
            if (!found) usage();
            have_mode = 1;
        } else if (!strcmp(a, "--out")) opt.out = v;
        else if (!strcmp(a, "--frames")) opt.frames = (uint32_t)parse_u64(v);
        else if (!strcmp(a, "--timing")) {
            if (!strcmp(v, "on")) opt.timing = 1;
            else if (!strcmp(v, "off")) opt.timing = 0;
            else usage();
        } else if (!strcmp(a, "--relative")) opt.relative_ns = parse_u64(v);
        else if (!strcmp(a, "--skip")) opt.skip = (uint32_t)parse_u64(v);
        else if (!strcmp(a, "--images")) opt.images = (uint32_t)parse_u64(v);
        else if (!strcmp(a, "--slice-us")) opt.slice_us = (uint32_t)parse_u64(v);
        else if (!strcmp(a, "--wait-lock")) {
            if (!strcmp(v, "on")) opt.wait_lock = 1;
            else if (!strcmp(v, "off")) opt.wait_lock = 0;
            else usage();
        }
        else usage();
    }
    if (!have_window || !have_mode || !opt.out) usage();
    if (opt.frames < 10 || opt.frames > 1000000 || opt.skip == 1 || opt.images < 2 || opt.images > MAX_IMAGES ||
        opt.slice_us < 50 || opt.slice_us > 100000)
        usage();
    if ((opt.relative_ns || opt.nearest) && !opt.timing) usage();
    if (opt.nearest && !opt.relative_ns) usage();
}

// ---- environment -------------------------------------------------------------------------------

typedef BOOL(WINAPI *PFN_EnumModules)(HANDLE, HMODULE *, DWORD, LPDWORD);

static void print_loaded_layers(void) {
    // Ground truth for the layers that wrap vkQueuePresentKHR: every DLL the loader brought in
    // that is not the loader, the driver or a system file.
    PFN_EnumModules enum_modules = (PFN_EnumModules)(void (*)(void))GetProcAddress(GetModuleHandleA("kernel32.dll"),
                                                                                    "K32EnumProcessModules");
    if (!enum_modules) { printf("layers error=no_K32EnumProcessModules\n"); return; }
    HMODULE mods[512];
    DWORD need = 0;
    enum_modules(GetCurrentProcess(), mods, sizeof mods, &need);
    unsigned n = need / sizeof(HMODULE), shown = 0;
    if (n > 512) n = 512;
    char sysdir[MAX_PATH];
    GetSystemDirectoryA(sysdir, sizeof sysdir);
    size_t syslen = strlen(sysdir);
    for (unsigned i = 0; i < n; i++) {
        char path[MAX_PATH];
        if (!GetModuleFileNameA(mods[i], path, sizeof path)) continue;
        if (!_strnicmp(path, sysdir, syslen)) continue;                       // loader, system DLLs
        if (strstr(path, "DriverStore") || strstr(path, "driverstore")) continue;   // the NVIDIA driver
        if (mods[i] == GetModuleHandleA(NULL)) continue;
        printf("layer_dll path=\"%s\"\n", path);
        shown++;
    }
    printf("layers loaded=%u\n", shown);
}

static void print_display(void) {
    DEVMODEW dm = {0};
    dm.dmSize = sizeof dm;
    if (EnumDisplaySettingsW(mon_device, ENUM_CURRENT_SETTINGS, &dm))
        printf("display width=%lu height=%lu hz=%lu bits=%lu\n", dm.dmPelsWidth, dm.dmPelsHeight,
               dm.dmDisplayFrequency, dm.dmBitsPerPel);
    // The desktop compositor's own refresh, from dwmapi (loaded at run time).
    HMODULE dwm = LoadLibraryA("dwmapi.dll");
    typedef DWM_TIMING_INFO TimingInfo;   // packed to 1 byte by dwmapi.h; cbSize must match
    typedef HRESULT(WINAPI * PFN_Timing)(HWND, TimingInfo *);
    PFN_Timing timing = dwm ? (PFN_Timing)(void (*)(void))GetProcAddress(dwm, "DwmGetCompositionTimingInfo") : NULL;
    if (timing) {
        TimingInfo ti;
        memset(&ti, 0, sizeof ti);
        ti.cbSize = sizeof ti;
        HRESULT hr = timing(NULL, &ti);
        printf("dwm hr=0x%08lx rate=%u/%u qpc_refresh_period=%llu\n", (unsigned long)hr, ti.rateRefresh.uiNumerator,
               ti.rateRefresh.uiDenominator, (unsigned long long)ti.qpcRefreshPeriod);
    }
}

// The driver's variable-refresh state of the primary display. Interface ids and the version-1
// record are NVIDIA's public NVAPI headers' (github.com/NVIDIA/nvapi, MIT): NvAPI_Initialize
// 0x0150e828, NvAPI_DISP_GetGDIPrimaryDisplayId 0x1e9d8a31, NvAPI_Disp_GetVRRInfo 0xdf8fda57.
static void print_vrr(const char *when) {
    typedef void *(*PFN_Query)(uint32_t);
    typedef struct { uint32_t version, bits, reserved[4]; } VrrInfo;
    HMODULE m = LoadLibraryA("nvapi64.dll");
    PFN_Query q = m ? (PFN_Query)(void (*)(void))GetProcAddress(m, "nvapi_QueryInterface") : NULL;
    if (!q) { printf("vrr when=%s nvapi=absent\n", when); return; }
    int (*init)(void) = (int (*)(void))q(0x0150e828u);
    int (*primary)(uint32_t *) = (int (*)(uint32_t *))q(0x1e9d8a31u);
    int (*info)(uint32_t, VrrInfo *) = (int (*)(uint32_t, VrrInfo *))q(0xdf8fda57u);
    if (!init || !primary || !info) { printf("vrr when=%s nvapi=incomplete\n", when); return; }
    uint32_t id = 0;
    int ri = init(), rp = primary(&id);
    VrrInfo v = {(uint32_t)sizeof(VrrInfo) | (1u << 16), 0, {0, 0, 0, 0}};
    int rv = info(id, &v);
    printf("vrr when=%s init=%d primary=%d display_id=0x%x result=%d enabled=%u possible=%u requested=%u "
           "indicator=%u in_vrr_mode=%u\n", when, ri, rp, id, rv, v.bits & 1, (v.bits >> 1) & 1, (v.bits >> 2) & 1,
           (v.bits >> 3) & 1, (v.bits >> 4) & 1);
}

static double tsc_hz(void) {
    int64_t q0 = qpc();
    uint64_t c0 = __rdtsc();
    Sleep(200);
    int64_t q1 = qpc();
    uint64_t c1 = __rdtsc();
    return (double)(c1 - c0) * (double)qpf / (double)(q1 - q0);
}

// ---- window ------------------------------------------------------------------------------------

static LRESULT CALLBACK wndproc(HWND h, UINT m, WPARAM w, LPARAM l) {
    switch (m) {
    case WM_CLOSE: InterlockedExchange(&window_lost, 1); return 0;
    case WM_SIZE: if (w == SIZE_MINIMIZED) InterlockedExchange(&window_lost, 1); break;
    case WM_PAINT: ValidateRect(h, NULL); return 0;
    case WM_ERASEBKGND: return 1;
    }
    return DefWindowProcW(h, m, w, l);
}

static void make_window(void) {
    typedef BOOL(WINAPI * PFN_Dpi)(HANDLE);
    PFN_Dpi dpi = (PFN_Dpi)(void (*)(void))GetProcAddress(GetModuleHandleA("user32.dll"),
                                                         "SetProcessDpiAwarenessContext");
    if (dpi) dpi((HANDLE)-4);   // DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2: sizes in physical pixels
    POINT origin = {0, 0};
    HMONITOR mon = MonitorFromPoint(origin, MONITOR_DEFAULTTOPRIMARY);
    MONITORINFOEXW mi;
    mi.cbSize = sizeof mi;
    if (!GetMonitorInfoW(mon, (MONITORINFO *)&mi)) die(2, "no primary monitor");
    mon_rect = mi.rcMonitor;
    memcpy(mon_device, mi.szDevice, sizeof mon_device);
    WNDCLASSW wc = {0};
    wc.lpfnWndProc = wndproc;
    wc.hInstance = GetModuleHandleW(NULL);
    wc.hCursor = LoadCursorW(NULL, (LPCWSTR)IDC_ARROW);
    wc.lpszClassName = L"vkpresent";
    RegisterClassW(&wc);
    int x, y, w, h;
    DWORD style;
    if (opt.borderless) {
        style = WS_POPUP;
        x = mon_rect.left; y = mon_rect.top;
        w = mon_rect.right - mon_rect.left; h = mon_rect.bottom - mon_rect.top;
    } else {
        style = WS_OVERLAPPEDWINDOW;
        RECT r = {0, 0, 1280, 720};
        AdjustWindowRect(&r, style, FALSE);
        x = mon_rect.left + 80; y = mon_rect.top + 80;
        w = r.right - r.left; h = r.bottom - r.top;
    }
    hwnd = CreateWindowExW(WS_EX_TOPMOST, L"vkpresent", L"vkpresent", style, x, y, w, h, NULL, NULL, wc.hInstance, NULL);
    if (!hwnd) die(2, "CreateWindowEx failed: %lu", GetLastError());
    ShowWindow(hwnd, SW_SHOW);
    SetWindowPos(hwnd, HWND_TOPMOST, x, y, w, h, SWP_SHOWWINDOW);
    SetForegroundWindow(hwnd);
    MSG msg;
    for (int i = 0; i < 20; i++) {
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        Sleep(10);
    }
    if (opt.occlude) {
        // A plain grey tool window over the corner, above the swapchain window, never activated.
        WNDCLASSW oc = {0};
        oc.lpfnWndProc = DefWindowProcW;
        oc.hInstance = wc.hInstance;
        oc.hbrBackground = (HBRUSH)(COLOR_BTNSHADOW + 1);
        oc.lpszClassName = L"vkpresent-occluder";
        RegisterClassW(&oc);
        POINT corner = {0, 0};
        ClientToScreen(hwnd, &corner);
        HWND occ = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE, L"vkpresent-occluder", L"",
                                   WS_POPUP, corner.x + 40, corner.y + 40, 240, 160, NULL, NULL, wc.hInstance, NULL);
        if (!occ) die(2, "the occluding window failed: %lu", GetLastError());
        ShowWindow(occ, SW_SHOWNOACTIVATE);
        SetWindowPos(occ, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW);
        for (int i = 0; i < 10; i++) {
            while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
            Sleep(10);
        }
    }
    RECT cr;
    GetClientRect(hwnd, &cr);
    printf("window kind=%s client_width=%ld client_height=%ld monitor_width=%ld monitor_height=%ld foreground=%d "
           "occluded=%d\n", opt.borderless ? "borderless" : "composed", cr.right, cr.bottom,
           mon_rect.right - mon_rect.left, mon_rect.bottom - mon_rect.top, GetForegroundWindow() == hwnd, opt.occlude);
}

// ---- Vulkan setup ------------------------------------------------------------------------------

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static void make_instance(void) {
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    const char *exts[] = {"VK_KHR_surface", "VK_KHR_win32_surface", "VK_KHR_get_surface_capabilities2"};
    VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "vkpresent";
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = 3;
    ci.ppEnabledExtensionNames = exts;
    VkResult r = vkCreateInstance(&ci, NULL, &inst);
    if (r != VK_SUCCESS) die(2, "vkCreateInstance failed: %d (no Vulkan 1.3 or no win32 surface)", (int)r);
    volkLoadInstance(inst);
    create_win32_surface = (PFN_vkCreateWin32SurfaceKHR)vkGetInstanceProcAddr(inst, "vkCreateWin32SurfaceKHR");
    if (!create_win32_surface) die(2, "no vkCreateWin32SurfaceKHR");
    const char *dis = getenv("VK_LOADER_LAYERS_DISABLE");
    printf("env layers_disable=\"%s\"\n", dis ? dis : "");
    uint32_t nl = 0;
    vkEnumerateInstanceLayerProperties(&nl, NULL);
    VkLayerProperties *lp = calloc(nl ? nl : 1, sizeof *lp);
    vkEnumerateInstanceLayerProperties(&nl, lp);
    for (uint32_t i = 0; i < nl; i++) printf("layer_listed name=%s\n", lp[i].layerName);
    free(lp);
    print_loaded_layers();
}

static void make_device(void) {
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (!n) die(2, "no Vulkan device");
    VkPhysicalDevice list[8];
    if (n > 8) n = 8;
    vkEnumeratePhysicalDevices(inst, &n, list);
    VkWin32SurfaceCreateInfoKHR sci = {VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR};
    sci.hinstance = GetModuleHandleW(NULL);
    sci.hwnd = hwnd;
    VKC(create_win32_surface(inst, &sci, NULL, &surface));
    for (uint32_t d = 0; d < n && family == UINT32_MAX; d++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(list[d], &p);
        uint32_t nq = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(list[d], &nq, NULL);
        VkQueueFamilyProperties qf[16];
        if (nq > 16) nq = 16;
        vkGetPhysicalDeviceQueueFamilyProperties(list[d], &nq, qf);
        for (uint32_t q = 0; q < nq; q++) {
            VkBool32 ok = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(list[d], q, surface, &ok);
            if (ok && (qf[q].queueFlags & VK_QUEUE_GRAPHICS_BIT)) { gpu = list[d]; family = q; break; }
        }
        if (family != UINT32_MAX)
            printf("device name=\"%s\" driver=%u.%u api=%u.%u.%u\n", p.deviceName, (p.driverVersion >> 22) & 0x3ff,
                   (p.driverVersion >> 14) & 0xff, VK_API_VERSION_MAJOR(p.apiVersion), VK_API_VERSION_MINOR(p.apiVersion),
                   VK_API_VERSION_PATCH(p.apiVersion));
    }
    if (family == UINT32_MAX) die(2, "no device can present to the window");

    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ne, NULL);
    VkExtensionProperties *ep = calloc(ne, sizeof *ep);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ne, ep);
    const char *want[] = {"VK_KHR_swapchain", "VK_KHR_present_id2", "VK_KHR_present_wait2",
                          "VK_KHR_calibrated_timestamps", "VK_EXT_present_timing",
                          "VK_KHR_present_mode_fifo_latest_ready"};
    const char *exts[6];
    uint32_t nexts = 0;
    for (int i = 0; i < 6; i++) {
        if (!has_ext(ep, ne, want[i])) {
            if (i == 5 && opt.mode != VK_PRESENT_MODE_FIFO_LATEST_READY_KHR) continue;
            die(2, "the device lacks %s", want[i]);
        }
        exts[nexts++] = want[i];
    }
    free(ep);

    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR latest = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR};
    VkPhysicalDevicePresentTimingFeaturesEXT timing = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
    VkPhysicalDevicePresentWait2FeaturesKHR wait2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR};
    VkPhysicalDevicePresentId2FeaturesKHR id2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
    VkPhysicalDeviceVulkan13Features v13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &v13; v13.pNext = &id2; id2.pNext = &wait2; wait2.pNext = &timing;
    timing.pNext = nexts == 6 ? &latest : NULL;
    vkGetPhysicalDeviceFeatures2(gpu, &f2);
    printf("features dynamic_rendering=%u present_id2=%u present_wait2=%u present_timing=%u at_absolute=%u "
           "at_relative=%u fifo_latest_ready=%u\n", v13.dynamicRendering, id2.presentId2, wait2.presentWait2,
           timing.presentTiming, timing.presentAtAbsoluteTime, timing.presentAtRelativeTime,
           latest.presentModeFifoLatestReady);
    if (!v13.dynamicRendering || !id2.presentId2 || !wait2.presentWait2 || !timing.presentTiming)
        die(2, "a required feature is missing");
    if (opt.relative_ns && !timing.presentAtRelativeTime) die(2, "presentAtRelativeTime is missing");
    if (opt.mode == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR && !latest.presentModeFifoLatestReady)
        die(2, "presentModeFifoLatestReady is missing");
    // Enable only what the run uses.
    VkPhysicalDeviceVulkan13Features e13 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    e13.dynamicRendering = VK_TRUE;
    VkPhysicalDevicePresentId2FeaturesKHR eid2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
    eid2.presentId2 = VK_TRUE;
    VkPhysicalDevicePresentWait2FeaturesKHR ew2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_WAIT_2_FEATURES_KHR};
    ew2.presentWait2 = VK_TRUE;
    VkPhysicalDevicePresentTimingFeaturesEXT et = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};
    et.presentTiming = VK_TRUE;
    et.presentAtRelativeTime = opt.relative_ns ? VK_TRUE : VK_FALSE;
    VkPhysicalDevicePresentModeFifoLatestReadyFeaturesKHR el = {
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_MODE_FIFO_LATEST_READY_FEATURES_KHR};
    el.presentModeFifoLatestReady = VK_TRUE;
    e13.pNext = &eid2; eid2.pNext = &ew2; ew2.pNext = &et;
    et.pNext = opt.mode == VK_PRESENT_MODE_FIFO_LATEST_READY_KHR ? &el : NULL;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &e13;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = nexts;
    dci.ppEnabledExtensionNames = exts;
    VKC(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    vkGetDeviceQueue(dev, family, 0, &queue);
}

static void make_swapchain(void) {
    VkPhysicalDeviceSurfaceInfo2KHR si = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR};
    si.surface = surface;
    VkPresentTimingSurfaceCapabilitiesEXT ptc = {VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
    VkSurfaceCapabilitiesPresentWait2KHR pw2 = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_WAIT_2_KHR};
    VkSurfaceCapabilitiesPresentId2KHR pid2 = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR};
    VkSurfaceCapabilities2KHR caps = {VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR};
    caps.pNext = &ptc; ptc.pNext = &pw2; pw2.pNext = &pid2;
    VKC(vkGetPhysicalDeviceSurfaceCapabilities2KHR(gpu, &si, &caps));
    VkSurfaceCapabilitiesKHR *c = &caps.surfaceCapabilities;
    printf("surface min_images=%u max_images=%u width=%u height=%u timing=%u at_absolute=%u at_relative=%u "
           "stages=0x%x present_id2=%u present_wait2=%u\n", c->minImageCount, c->maxImageCount, c->currentExtent.width,
           c->currentExtent.height, ptc.presentTimingSupported, ptc.presentAtAbsoluteTimeSupported,
           ptc.presentAtRelativeTimeSupported, ptc.presentStageQueries, pid2.presentId2Supported,
           pw2.presentWait2Supported);
    if (!pid2.presentId2Supported || !pw2.presentWait2Supported) die(2, "the surface lacks present id 2 or present wait 2");
    if (opt.timing && !ptc.presentTimingSupported) die(2, "the surface lacks present timing");
    if (opt.relative_ns && !ptc.presentAtRelativeTimeSupported) die(2, "the surface lacks relative-time presents");
    uint32_t nm = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &nm, NULL);
    VkPresentModeKHR modes[16];
    if (nm > 16) nm = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(gpu, surface, &nm, modes);
    int ok = 0;
    for (uint32_t i = 0; i < nm; i++) ok |= modes[i] == opt.mode;
    if (!ok) die(2, "the surface does not offer present mode %s", opt.mode_name);

    extent = c->currentExtent;
    if (extent.width == UINT32_MAX || !extent.width) die(1, "the window has no size");
    uint32_t count = opt.images < c->minImageCount ? c->minImageCount : opt.images;
    if (c->maxImageCount && count > c->maxImageCount) count = c->maxImageCount;
    VkSwapchainCreateInfoKHR ci = {VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.flags = VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR | VK_SWAPCHAIN_CREATE_PRESENT_WAIT_2_BIT_KHR |
               (opt.timing ? VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT : 0);
    ci.surface = surface;
    ci.minImageCount = count;
    ci.imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    ci.imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = c->currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = opt.mode;
    ci.clipped = VK_TRUE;
    VKC(vkCreateSwapchainKHR(dev, &ci, NULL, &swapchain));
    nimages = MAX_IMAGES;
    VKC(vkGetSwapchainImagesKHR(dev, swapchain, &nimages, images));
    for (uint32_t i = 0; i < nimages; i++) {
        VkImageViewCreateInfo vi = {VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = images[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_B8G8R8A8_UNORM;
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.layerCount = 1;
        VKC(vkCreateImageView(dev, &vi, NULL, &views[i]));
        VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VKC(vkCreateSemaphore(dev, &sci, NULL, &done_sem[i]));
    }
    printf("swapchain mode=%s images=%u requested_images=%u width=%u height=%u timing=%d relative_ns=%llu nearest=%d "
           "wait_lock=%d slice_us=%u\n", opt.mode_name, nimages, opt.images, extent.width, extent.height, opt.timing,
           (unsigned long long)opt.relative_ns, opt.nearest, opt.wait_lock, opt.slice_us);

    if (opt.timing) {
        VKC(vkSetSwapchainPresentTimingQueueSizeEXT(dev, swapchain, 256));
        uint64_t counter = 0;
        VkSwapchainTimeDomainPropertiesEXT dp = {VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT};
        VKC(vkGetSwapchainTimeDomainPropertiesEXT(dev, swapchain, &dp, &counter));
        VkTimeDomainKHR kinds[MAX_DOMAINS];
        uint64_t ids[MAX_DOMAINS];
        if (dp.timeDomainCount > MAX_DOMAINS) dp.timeDomainCount = MAX_DOMAINS;
        dp.pTimeDomains = kinds;
        dp.pTimeDomainIds = ids;
        VkResult r = vkGetSwapchainTimeDomainPropertiesEXT(dev, swapchain, &dp, &counter);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) die(3, "vkGetSwapchainTimeDomainPropertiesEXT failed: %d", (int)r);
        // Prefer the performance counter (no calibration needed), then a swapchain-local domain.
        int best = -1, rank = 99;
        for (uint32_t i = 0; i < dp.timeDomainCount; i++) {
            printf("time_domain kind=%d id=%llu\n", (int)kinds[i], (unsigned long long)ids[i]);
            int k = kinds[i] == VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR ? 0
                  : kinds[i] == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT        ? 1
                  : kinds[i] == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT    ? 2 : 3;
            if (k < rank) { rank = k; best = (int)i; }
        }
        if (best < 0) die(2, "the swapchain lists no time domain");
        domain_id = ids[best];
        domain_kind = kinds[best];
        printf("time_domain_chosen kind=%d id=%llu\n", (int)domain_kind, (unsigned long long)domain_id);
    }
}

static void print_timing_props(const char *when) {
    if (!opt.timing) return;
    VkSwapchainTimingPropertiesEXT tp = {VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT};
    uint64_t counter = 0;
    VkResult r = vkGetSwapchainTimingPropertiesEXT(dev, swapchain, &tp, &counter);
    if (r == VK_SUCCESS && counter == last_props_counter) return;
    if (r == VK_SUCCESS) last_props_counter = counter;
    printf("timing_props when=%s result=%d counter=%llu refresh_duration_ns=%llu refresh_interval_ns=%llu qpc=%lld\n",
           when, (int)r, (unsigned long long)counter, (unsigned long long)tp.refreshDuration,
           (unsigned long long)tp.refreshInterval, (long long)qpc());
}

// One calibration pair: the domain's value and the performance counter at the same instant.
static void calibrate(VkTimeDomainKHR kind, uint64_t id, VkPresentStageFlagsEXT stage, const char *when) {
    VkSwapchainCalibratedTimestampInfoEXT sc = {VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT};
    sc.swapchain = swapchain;
    sc.presentStage = kind == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ? stage : 0;
    sc.timeDomainId = id;
    VkCalibratedTimestampInfoKHR info[2] = {{VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR},
                                            {VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR}};
    info[0].timeDomain = kind;
    info[0].pNext = (kind == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT || kind == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT)
                        ? &sc : NULL;
    info[1].timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR;
    uint64_t ts[2] = {0, 0}, dev_ns = 0;
    VkResult r = vkGetCalibratedTimestampsKHR(dev, 2, info, ts, &dev_ns);
    printf("calib when=%s kind=%d id=%llu stage=%u result=%d value=%llu qpc=%llu max_deviation_ns=%llu\n", when,
           (int)kind, (unsigned long long)id, (unsigned)stage, (int)r, (unsigned long long)ts[0],
           (unsigned long long)ts[1], (unsigned long long)dev_ns);
}

static void calibrate_seen(const char *when) {
    for (uint32_t i = 0; i < nseen; i++) {
        calibrate(seen_domains[i].kind, seen_domains[i].id, VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT, when);
        if (seen_domains[i].kind == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT)
            calibrate(seen_domains[i].kind, seen_domains[i].id, VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT, when);
    }
}

// ---- present timing results --------------------------------------------------------------------

static uint32_t timing_results, timing_incomplete_seen;

static void poll_timing(void) {
    if (!opt.timing) return;
    static VkPastPresentationTimingEXT arr[TIMING_BATCH];
    static VkPresentStageTimeEXT stages[TIMING_BATCH][4];
    for (;;) {
        for (int i = 0; i < TIMING_BATCH; i++) {
            memset(&arr[i], 0, sizeof arr[i]);
            arr[i].sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT;
            arr[i].presentStageCount = 4;
            arr[i].pPresentStages = stages[i];
        }
        VkPastPresentationTimingInfoEXT info = {VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT};
        info.swapchain = swapchain;
        VkPastPresentationTimingPropertiesEXT props = {VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT};
        props.presentationTimingCount = TIMING_BATCH;
        props.pPresentationTimings = arr;
        VkResult r = vkGetPastPresentationTimingEXT(dev, &info, &props);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) { printf("timing_poll result=%d\n", (int)r); return; }
        for (uint32_t i = 0; i < props.presentationTimingCount; i++) {
            VkPastPresentationTimingEXT *t = &arr[i];
            if (!t->presentId || t->presentId > opt.frames) continue;
            Rec *e = &rec[t->presentId - 1];
            uint32_t d = 0;
            while (d < nseen && !(seen_domains[d].kind == t->timeDomain && seen_domains[d].id == t->timeDomainId)) d++;
            if (d == nseen && nseen < MAX_DOMAINS) {
                seen_domains[nseen].kind = t->timeDomain;
                seen_domains[nseen].id = t->timeDomainId;
                nseen++;
                calibrate_seen("first");
            }
            e->pt_domain = (uint8_t)d;
            e->pt_complete = t->reportComplete ? 1 : 0;
            if (!t->reportComplete) timing_incomplete_seen++;
            for (uint32_t s = 0; s < t->presentStageCount && s < 4; s++) {
                if (t->pPresentStages[s].stage == VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT) {
                    e->pt_qoe = t->pPresentStages[s].time;
                    e->pt_have |= 1;
                } else if (t->pPresentStages[s].stage == VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) {
                    e->pt_fpo = t->pPresentStages[s].time;
                    e->pt_have |= 2;
                }
            }
            timing_results++;
        }
        if (r == VK_SUCCESS) return;
    }
}

// ---- threads -----------------------------------------------------------------------------------

static void sleep_until(HANDLE timer, int64_t when) {
    int64_t now = qpc();
    int64_t margin = qpf / 1000;   // the last millisecond is a spin
    if (when - now > margin + qpf / 2000) {
        LARGE_INTEGER due;
        due.QuadPart = -(LONGLONG)((when - now - margin) * 10000000 / qpf);   // 100 ns units, relative
        SetWaitableTimer(timer, &due, 0, NULL, NULL, FALSE);
        WaitForSingleObject(timer, INFINITE);
    }
    while (qpc() < when) _mm_pause();
}

static DWORD WINAPI producer_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    HANDLE timer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /* CREATE_WAITABLE_TIMER_HIGH_RESOLUTION */,
                                          TIMER_ALL_ACCESS);
    if (!timer) timer = CreateWaitableTimerW(NULL, FALSE, NULL);
    for (uint32_t n = 0; n < opt.frames && !window_lost && !failed_code; n++) {
        int64_t due = t0 + (int64_t)(((__int128)n * qpf + TICK_HZ - 1) / TICK_HZ);
        sleep_until(timer, due);
        rec[n].due = due;
        rec[n].produced = qpc();
        if (opt.skip && n % opt.skip == opt.skip - 1) { rec[n].state = ST_SKIPPED; continue; }
        rec[n].state = ST_PRODUCED;
        InterlockedExchange(&last_produced, (LONG)n);
        InterlockedExchange(&mailbox, (LONG)(n + 1));
        SetEvent(ev_frame);
    }
    InterlockedExchange(&producer_done, 1);
    SetEvent(ev_frame);
    CloseHandle(timer);
    return 0;
}

static VkCommandPool pool;
static VkCommandBuffer cmd[IN_FLIGHT];
static VkFence fence[IN_FLIGHT];
static VkSemaphore acq_sem[IN_FLIGHT];

static void record(VkCommandBuffer cb, uint32_t idx, uint32_t n) {
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKC(vkBeginCommandBuffer(cb, &bi));
    VkImageMemoryBarrier b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = images[idx];
    b.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    b.subresourceRange.levelCount = 1;
    b.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
    VkRenderingAttachmentInfo att = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    att.imageView = views[idx];
    att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.clearValue.color.float32[0] = att.clearValue.color.float32[1] = att.clearValue.color.float32[2] = 0.06f;
    att.clearValue.color.float32[3] = 1.0f;
    VkRenderingInfo ri = {VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea.extent = extent;
    ri.layerCount = 1;
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &att;
    vkCmdBeginRendering(cb, &ri);
    // A bar that moves 8 pixels a tick, so a repeated or skipped tick is visible to an eye too.
    uint32_t bw = 48, span = extent.width > bw ? extent.width - bw : 1;
    VkClearAttachment ca = {VK_IMAGE_ASPECT_COLOR_BIT, 0, {{{0.30f, 0.30f, 0.30f, 1.0f}}}};
    VkClearRect cr = {{{(int32_t)((n * 8u) % span), 0}, {bw, extent.height}}, 0, 1};
    vkCmdClearAttachments(cb, 1, &ca, 1, &cr);
    vkCmdEndRendering(cb);
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                         NULL, 0, NULL, 1, &b);
    VKC(vkEndCommandBuffer(cb));
}

static DWORD WINAPI present_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    uint32_t f = 0;
    int64_t last = -1;
    for (;;) {
        // An auto-reset event folds two signals into one: the producer's last frame and its end can
        // arrive while this thread is busy, so the wait is bounded and the end is checked again.
        WaitForSingleObject(ev_frame, 50);
        LONG got = InterlockedExchange(&mailbox, 0);
        if (!got) {
            if (producer_done || window_lost || failed_code) break;
            continue;
        }
        uint32_t n = (uint32_t)(got - 1);
        Rec *e = &rec[n];
        ULONG64 c0 = 0, c1 = 0;
        QueryThreadCycleTime(GetCurrentThread(), &c0);
        e->wake = qpc();
        for (int64_t k = last + 1; k < (int64_t)n; k++)
            if (rec[k].state == ST_PRODUCED) rec[k].state = ST_HOST_DROPPED;
        last = n;
        InterlockedExchange(&present_step, 1);
        VKC(vkWaitForFences(dev, 1, &fence[f], VK_TRUE, UINT64_MAX));
        VKC(vkResetFences(dev, 1, &fence[f]));
        uint32_t idx = 0;
        InterlockedExchange(&present_step, 2);
        VkResult r = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acq_sem[f], VK_NULL_HANDLE, &idx);
        e->acquired = qpc();
        e->acq_res = r;
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            printf("acquire_failed tick=%u result=%d\n", n, (int)r);
            InterlockedExchange((volatile LONG *)&failed_code, 1);
            break;
        }
        e->image = idx;
        InterlockedExchange(&present_step, 3);
        record(cmd[f], idx, n);
        VkPipelineStageFlags ws = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.waitSemaphoreCount = 1;
        si.pWaitSemaphores = &acq_sem[f];
        si.pWaitDstStageMask = &ws;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd[f];
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &done_sem[idx];
        VKC(vkQueueSubmit(queue, 1, &si, fence[f]));
        uint64_t id = (uint64_t)n + 1;
        VkPresentId2KHR pid = {VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR};
        pid.swapchainCount = 1;
        pid.pPresentIds = &id;
        VkPresentTimingInfoEXT ti = {VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT};
        ti.presentStageQueries = VK_PRESENT_STAGE_QUEUE_OPERATIONS_END_BIT_EXT | VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT;
        ti.timeDomainId = domain_id;
        if (opt.relative_ns) {
            ti.flags = VK_PRESENT_TIMING_INFO_PRESENT_AT_RELATIVE_TIME_BIT_EXT |
                       (opt.nearest ? VK_PRESENT_TIMING_INFO_PRESENT_AT_NEAREST_REFRESH_CYCLE_BIT_EXT : 0);
            ti.targetTime = opt.relative_ns;
        }
        VkPresentTimingsInfoEXT tis = {VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT};
        tis.swapchainCount = 1;
        tis.pTimingInfos = &ti;
        if (opt.timing) pid.pNext = &tis;
        VkPresentInfoKHR pi = {VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.pNext = &pid;
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &done_sem[idx];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &idx;
        int64_t lock0 = qpc();
        InterlockedExchange(&present_step, 4);
        InterlockedExchange(&present_wants_lock, 1);
        AcquireSRWLockExclusive(&sc_lock);
        InterlockedExchange(&present_wants_lock, 0);
        e->present_call = qpc();
        e->lock_wait = e->present_call - lock0;
        InterlockedExchange(&present_step, 5);
        r = vkQueuePresentKHR(queue, &pi);
        InterlockedExchange(&present_step, 0);
        InterlockedExchange(&last_presented, (LONG)n);
        e->present_ret = qpc();
        ReleaseSRWLockExclusive(&sc_lock);
        QueryThreadCycleTime(GetCurrentThread(), &c1);
        e->cycles = c1 - c0;
        e->pres_res = r;
        if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) {
            printf("present_failed tick=%u result=%d\n", n, (int)r);
            InterlockedExchange((volatile LONG *)&failed_code, 1);
            break;
        }
        e->state = ST_PRESENTED;
        presented_frames++;
        pushed[npushed] = id;
        InterlockedIncrement(&npushed);
        SetEvent(ev_pushed);
        f = (f + 1) % IN_FLIGHT;
    }
    FILETIME a, b, k, u;
    GetThreadTimes(GetCurrentThread(), &a, &b, &k, &u);
    present_thread_kernel100ns = (int64_t)(((uint64_t)k.dwHighDateTime << 32) | k.dwLowDateTime);
    present_thread_user100ns = (int64_t)(((uint64_t)u.dwHighDateTime << 32) | u.dwLowDateTime);
    InterlockedExchange(&present_done, 1);
    SetEvent(ev_pushed);
    return 0;
}

static DWORD WINAPI wait_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    LONG done = 0;
    uint32_t since_props = 0;
    for (;;) {
        if (done >= npushed) {
            if (present_done && done >= npushed) break;
            WaitForSingleObject(ev_pushed, 20);
            continue;
        }
        uint64_t id = pushed[done];
        VkPresentWait2InfoKHR wi = {VK_STRUCTURE_TYPE_PRESENT_WAIT_2_INFO_KHR};
        wi.presentId = id;
        wi.timeout = (uint64_t)opt.slice_us * 1000;
        VkResult r = VK_TIMEOUT;
        int64_t start = qpc();
        if (!opt.wait_lock) {
            wi.timeout = 2000000000ull;
            r = vkWaitForPresent2KHR(dev, swapchain, &wi);
            if (r != VK_TIMEOUT) rec[id - 1].wait_ret = qpc();
        }
        while (r == VK_TIMEOUT && opt.wait_lock) {
            AcquireSRWLockExclusive(&sc_lock);
            r = vkWaitForPresent2KHR(dev, swapchain, &wi);
            int64_t now = qpc();
            ReleaseSRWLockExclusive(&sc_lock);
            if (r != VK_TIMEOUT) { rec[id - 1].wait_ret = now; break; }
            if (now - start > 2 * qpf) break;   // never shown within 2 s
            while (present_wants_lock) SwitchToThread();
        }
        rec[id - 1].wait_res = r;
        InterlockedExchange(&last_waited, (LONG)(id - 1));
        done++;
        poll_timing();
        if (++since_props >= 60) {
            since_props = 0;
            AcquireSRWLockExclusive(&sc_lock);
            print_timing_props("run");
            ReleaseSRWLockExclusive(&sc_lock);
        }
    }
    return 0;
}

// The display's own vertical blanks, through the kernel thunks of gdi32 (no swapchain involved).
typedef struct { HDC hdc; UINT adapter; LUID luid; UINT source; } OpenFromHdc;
typedef struct { UINT adapter; UINT device; UINT source; } WaitVblank;
typedef struct { UINT adapter; } CloseAdapter;
typedef LONG(APIENTRY *PFN_Open)(OpenFromHdc *);
typedef LONG(APIENTRY *PFN_Wait)(const WaitVblank *);
typedef LONG(APIENTRY *PFN_Close)(const CloseAdapter *);
static volatile LONG vblank_stop;

static DWORD WINAPI vblank_main(void *arg) {
    (void)arg;
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    HMODULE gdi = LoadLibraryA("gdi32.dll");
    typedef HDC(WINAPI * PFN_CreateDC)(LPCWSTR, LPCWSTR, LPCWSTR, const DEVMODEW *);
    typedef BOOL(WINAPI * PFN_DeleteDC)(HDC);
    PFN_CreateDC create_dc = gdi ? (PFN_CreateDC)(void (*)(void))GetProcAddress(gdi, "CreateDCW") : NULL;
    PFN_DeleteDC delete_dc = gdi ? (PFN_DeleteDC)(void (*)(void))GetProcAddress(gdi, "DeleteDC") : NULL;
    PFN_Open open_ad = gdi ? (PFN_Open)(void (*)(void))GetProcAddress(gdi, "D3DKMTOpenAdapterFromHdc") : NULL;
    PFN_Wait wait_vb = gdi ? (PFN_Wait)(void (*)(void))GetProcAddress(gdi, "D3DKMTWaitForVerticalBlankEvent") : NULL;
    PFN_Close close_ad = gdi ? (PFN_Close)(void (*)(void))GetProcAddress(gdi, "D3DKMTCloseAdapter") : NULL;
    if (!create_dc || !open_ad || !wait_vb) { printf("vblank error=no_d3dkmt\n"); return 0; }
    HDC hdc = create_dc(L"DISPLAY", mon_device, NULL, NULL);
    OpenFromHdc o = {hdc, 0, {0, 0}, 0};
    LONG st = open_ad(&o);
    if (st) { printf("vblank error=open status=0x%08lx\n", (unsigned long)st); return 0; }
    WaitVblank w = {o.adapter, 0, o.source};
    while (!vblank_stop) {
        if (wait_vb(&w)) { printf("vblank error=wait\n"); break; }
        int64_t t = qpc();
        LONG i = nvblanks;
        if ((uint32_t)i < max_vblanks) { vblanks[i] = t; InterlockedIncrement(&nvblanks); }
    }
    CloseAdapter c = {o.adapter};
    if (close_ad) close_ad(&c);
    if (delete_dc) delete_dc(hdc);
    return 0;
}

// ---- output ------------------------------------------------------------------------------------

static void write_outputs(void) {
    char path[1024];
    snprintf(path, sizeof path, "%s.frames.tsv", opt.out);
    FILE *f = fopen(path, "w");
    if (!f) die(1, "cannot write %s", path);
    fprintf(f, "tick\tstate\tdue\tproduced\twake\tacquired\tpresent_call\tpresent_ret\twait_ret\tlock_wait\t"
               "cycles\tpt_have\tpt_complete\tpt_domain\tpt_qoe\tpt_fpo\tacq_res\tpres_res\twait_res\timage\n");
    for (uint32_t n = 0; n < opt.frames; n++) {
        Rec *e = &rec[n];
        fprintf(f, "%u\t%u\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%llu\t%u\t%u\t%u\t%llu\t%llu\t%d\t%d\t%d\t%u\n", n,
                e->state, (long long)e->due, (long long)e->produced, (long long)e->wake, (long long)e->acquired,
                (long long)e->present_call, (long long)e->present_ret, (long long)e->wait_ret, (long long)e->lock_wait,
                (unsigned long long)e->cycles, e->pt_have, e->pt_complete, e->pt_domain,
                (unsigned long long)e->pt_qoe, (unsigned long long)e->pt_fpo, e->acq_res, e->pres_res, e->wait_res,
                e->image);
    }
    fclose(f);
    snprintf(path, sizeof path, "%s.vblanks.tsv", opt.out);
    f = fopen(path, "w");
    if (!f) die(1, "cannot write %s", path);
    fprintf(f, "qpc\n");
    for (LONG i = 0; i < nvblanks; i++) fprintf(f, "%lld\n", (long long)vblanks[i]);
    fclose(f);
}

int main(int argc, char **argv) {
    parse_args(argc, argv);
    setvbuf(stdout, NULL, _IOLBF, 0);
    LARGE_INTEGER fq;
    QueryPerformanceFrequency(&fq);
    qpf = fq.QuadPart;
    rec = calloc(opt.frames, sizeof *rec);
    pushed = calloc(opt.frames, sizeof *pushed);
    max_vblanks = opt.frames * 4 + 4096;
    vblanks = calloc(max_vblanks, sizeof *vblanks);
    if (!rec || !pushed || !vblanks) die(1, "out of memory");
    printf("start qpf=%lld frames=%u skip=%u tick_hz=%d\n", (long long)qpf, opt.frames, opt.skip, TICK_HZ);
    printf("tsc hz=%.0f\n", tsc_hz());

    make_window();
    print_display();
    make_instance();
    make_device();
    make_swapchain();

    VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    VKC(vkCreateCommandPool(dev, &pci, NULL, &pool));
    VkCommandBufferAllocateInfo alloc = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    alloc.commandPool = pool;
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandBufferCount = IN_FLIGHT;
    VKC(vkAllocateCommandBuffers(dev, &alloc, cmd));
    for (int i = 0; i < IN_FLIGHT; i++) {
        VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        VKC(vkCreateFence(dev, &fci, NULL, &fence[i]));
        VkSemaphoreCreateInfo sci = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VKC(vkCreateSemaphore(dev, &sci, NULL, &acq_sem[i]));
    }
    print_timing_props("start");
    print_vrr("start");

    ev_frame = CreateEventW(NULL, FALSE, FALSE, NULL);
    ev_pushed = CreateEventW(NULL, FALSE, FALSE, NULL);
    HANDLE vb = CreateThread(NULL, 0, vblank_main, NULL, 0, NULL);
    t0 = qpc() + qpf / 2;
    printf("clock t0=%lld\n", (long long)t0);
    HANDLE th[3];
    th[0] = CreateThread(NULL, 0, wait_main, NULL, 0, NULL);
    th[1] = CreateThread(NULL, 0, present_main, NULL, 0, NULL);
    th[2] = CreateThread(NULL, 0, producer_main, NULL, 0, NULL);
    // The window's thread pumps messages until the three threads end.
    int vrr_mid = 0;
    LONG seen_presented = -2, seen_waited = -2;
    int64_t moved = qpc();
    while (WaitForMultipleObjects(3, th, TRUE, 0) == WAIT_TIMEOUT) {
        MsgWaitForMultipleObjects(3, th, FALSE, 100, QS_ALLINPUT);
        if (last_presented != seen_presented || last_waited != seen_waited) {
            seen_presented = last_presented;
            seen_waited = last_waited;
            moved = qpc();
        } else if (qpc() - moved > 5 * qpf) {
            printf("stall produced=%ld presented=%ld waited=%ld present_step=%ld pushed=%ld timing_results=%u\n",
                   (long)last_produced, (long)last_presented, (long)last_waited, (long)present_step, (long)npushed,
                   timing_results);
            write_outputs();
            fflush(stdout);
            ExitProcess(4);
        }
        if (!vrr_mid && (uint32_t)npushed >= opt.frames / 2) { vrr_mid = 1; print_vrr("mid"); }
        MSG msg;
        while (PeekMessageW(&msg, NULL, 0, 0, PM_REMOVE)) { TranslateMessage(&msg); DispatchMessageW(&msg); }
        if (window_lost) SetEvent(ev_frame);
    }
    vblank_stop = 1;
    WaitForSingleObject(vb, 1000);
    vkDeviceWaitIdle(dev);
    poll_timing();
    print_timing_props("end");
    calibrate_seen("end");

    uint32_t produced = 0, skipped = 0, host_dropped = 0;
    for (uint32_t n = 0; n < opt.frames; n++) {
        produced += rec[n].state >= ST_PRODUCED;
        skipped += rec[n].state == ST_SKIPPED;
        host_dropped += rec[n].state == ST_HOST_DROPPED;
    }
    write_outputs();
    printf("result frames=%u produced=%u skipped=%u host_dropped=%u presented=%u timing_results=%u "
           "timing_incomplete_seen=%u vblanks=%ld present_thread_user_ms=%.3f present_thread_kernel_ms=%.3f "
           "window_lost=%ld failed=%d\n", opt.frames, produced, skipped, host_dropped, presented_frames, timing_results,
           timing_incomplete_seen, (long)nvblanks, present_thread_user100ns / 1e4, present_thread_kernel100ns / 1e4,
           (long)window_lost, failed_code);
    vkDestroySwapchainKHR(dev, swapchain, NULL);
    DestroyWindow(hwnd);
    return window_lost || failed_code ? 1 : 0;
}
