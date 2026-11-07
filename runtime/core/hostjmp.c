// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The host half of RT_SETJMP and rt_longjmp: a setjmp that saves every callee-saved register of the host ABI and a
// longjmp that restores them without unwinding, the way the C library's do when no unwind is asked for. Clang's
// __builtin_setjmp restores only the frame and stack pointers, and the boundary test caught a value the compiler kept
// in a callee-saved register coming back wrong after the jump; a function declared returns_twice that saves all of
// them is the C contract (an object not modified after the setjmp keeps its value).
//
// Buffer layout, in uint64_t words (RtJmp.host, 32 words):
//   Windows x64: rbx rbp rdi rsi r12 r13 r14 r15, rsp after return, return address, xmm6..xmm15 (bytes 80..239)
//   System V:    rbx rbp r12 r13 r14 r15, rsp after return, return address
//
// No unwind runs across the frames the jump abandons: lifted code has no destructors or SEH handlers to run. A process
// marked for hardware shadow stacks would refuse the jump; the program is not marked (CET compatibility is opt-in).
#include "oot2/cpu.h"

#ifdef _WIN32
__asm__(
    ".text\n"
    ".globl rt_host_setjmp\n"
    ".p2align 4\n"
    "rt_host_setjmp:\n"
    "    movq %rbx, 0(%rcx)\n"
    "    movq %rbp, 8(%rcx)\n"
    "    movq %rdi, 16(%rcx)\n"
    "    movq %rsi, 24(%rcx)\n"
    "    movq %r12, 32(%rcx)\n"
    "    movq %r13, 40(%rcx)\n"
    "    movq %r14, 48(%rcx)\n"
    "    movq %r15, 56(%rcx)\n"
    "    leaq 8(%rsp), %rdx\n"
    "    movq %rdx, 64(%rcx)\n"
    "    movq (%rsp), %rdx\n"
    "    movq %rdx, 72(%rcx)\n"
    "    movdqu %xmm6, 80(%rcx)\n"
    "    movdqu %xmm7, 96(%rcx)\n"
    "    movdqu %xmm8, 112(%rcx)\n"
    "    movdqu %xmm9, 128(%rcx)\n"
    "    movdqu %xmm10, 144(%rcx)\n"
    "    movdqu %xmm11, 160(%rcx)\n"
    "    movdqu %xmm12, 176(%rcx)\n"
    "    movdqu %xmm13, 192(%rcx)\n"
    "    movdqu %xmm14, 208(%rcx)\n"
    "    movdqu %xmm15, 224(%rcx)\n"
    "    xorl %eax, %eax\n"
    "    ret\n"
    ".globl rt_host_longjmp\n"
    ".p2align 4\n"
    "rt_host_longjmp:\n"
    "    movq 0(%rcx), %rbx\n"
    "    movq 8(%rcx), %rbp\n"
    "    movq 16(%rcx), %rdi\n"
    "    movq 24(%rcx), %rsi\n"
    "    movq 32(%rcx), %r12\n"
    "    movq 40(%rcx), %r13\n"
    "    movq 48(%rcx), %r14\n"
    "    movq 56(%rcx), %r15\n"
    "    movdqu 80(%rcx), %xmm6\n"
    "    movdqu 96(%rcx), %xmm7\n"
    "    movdqu 112(%rcx), %xmm8\n"
    "    movdqu 128(%rcx), %xmm9\n"
    "    movdqu 144(%rcx), %xmm10\n"
    "    movdqu 160(%rcx), %xmm11\n"
    "    movdqu 176(%rcx), %xmm12\n"
    "    movdqu 192(%rcx), %xmm13\n"
    "    movdqu 208(%rcx), %xmm14\n"
    "    movdqu 224(%rcx), %xmm15\n"
    "    movl %edx, %eax\n"
    "    movq 64(%rcx), %rsp\n"
    "    jmpq *72(%rcx)\n");
#else
__asm__(
    ".text\n"
    ".globl rt_host_setjmp\n"
    ".type rt_host_setjmp, @function\n"
    ".p2align 4\n"
    "rt_host_setjmp:\n"
    "    movq %rbx, 0(%rdi)\n"
    "    movq %rbp, 8(%rdi)\n"
    "    movq %r12, 16(%rdi)\n"
    "    movq %r13, 24(%rdi)\n"
    "    movq %r14, 32(%rdi)\n"
    "    movq %r15, 40(%rdi)\n"
    "    leaq 8(%rsp), %rdx\n"
    "    movq %rdx, 48(%rdi)\n"
    "    movq (%rsp), %rdx\n"
    "    movq %rdx, 56(%rdi)\n"
    "    xorl %eax, %eax\n"
    "    ret\n"
    ".size rt_host_setjmp, .-rt_host_setjmp\n"
    ".globl rt_host_longjmp\n"
    ".type rt_host_longjmp, @function\n"
    ".p2align 4\n"
    "rt_host_longjmp:\n"
    "    movq 0(%rdi), %rbx\n"
    "    movq 8(%rdi), %rbp\n"
    "    movq 16(%rdi), %r12\n"
    "    movq 24(%rdi), %r13\n"
    "    movq 32(%rdi), %r14\n"
    "    movq 40(%rdi), %r15\n"
    "    movl %esi, %eax\n"
    "    movq 48(%rdi), %rsp\n"
    "    jmpq *56(%rdi)\n"
    ".size rt_host_longjmp, .-rt_host_longjmp\n"
    ".section .note.GNU-stack,\"\",@progbits\n");
#endif
