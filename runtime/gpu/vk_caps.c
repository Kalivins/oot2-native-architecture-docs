// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Which SPIR-V capabilities and extensions the required list gives (oot2/vk_caps.h). The table pairs every capability
// the build names (spv_capability_name) that Vulkan can offer with the feature behind it, as the Vulkan specification's
// SPIR-V capability table gives it; a capability the table has no row for is not in the list. The features are read
// from the structure oot2_vk_enable fills, that is, from what the device is created with, so the answer follows the
// items and nothing else.
#include <stdio.h>
#include <string.h>

#include "oot2/vk_caps.h"

// CAP(value, expression over `e`, the features the device is created with)
#define VKC_CAPS(CAP)                                                                                                  \
    CAP(0, 1) /* Matrix */                                                                                             \
    CAP(1, 1) /* Shader */                                                                                             \
    CAP(9, e->f12.shaderFloat16)                                                                                       \
    CAP(10, e->features.shaderFloat64)                                                                                 \
    CAP(11, e->features.shaderInt64)                                                                                   \
    CAP(12, e->f12.shaderBufferInt64Atomics)                                                                           \
    CAP(22, e->features.shaderInt16)                                                                                   \
    CAP(25, e->features.shaderImageGatherExtended)                                                                     \
    CAP(27, e->features.shaderStorageImageMultisample)                                                                 \
    CAP(28, e->features.shaderUniformBufferArrayDynamicIndexing)                                                       \
    CAP(29, e->features.shaderSampledImageArrayDynamicIndexing)                                                        \
    CAP(30, e->features.shaderStorageBufferArrayDynamicIndexing)                                                       \
    CAP(31, e->features.shaderStorageImageArrayDynamicIndexing)                                                        \
    CAP(32, e->features.shaderClipDistance)                                                                            \
    CAP(33, e->features.shaderCullDistance)                                                                            \
    CAP(34, e->features.imageCubeArray)                                                                                \
    CAP(35, e->features.sampleRateShading)                                                                             \
    CAP(39, e->f12.shaderInt8)                                                                                         \
    CAP(40, 1)                                                                                                         \
    CAP(41, e->features.shaderResourceResidency)                                                                       \
    CAP(42, e->features.shaderResourceMinLod)                                                                          \
    CAP(43, 1) /* Sampled1D */                                                                                         \
    CAP(44, 1) /* Image1D */                                                                                           \
    CAP(45, e->features.imageCubeArray)                                                                                \
    CAP(46, 1) /* SampledBuffer */                                                                                     \
    CAP(47, 1) /* ImageBuffer */                                                                                       \
    CAP(48, e->features.shaderStorageImageMultisample)                                                                 \
    CAP(49, e->features.shaderStorageImageExtendedFormats)                                                             \
    CAP(50, 1) /* ImageQuery */                                                                                        \
    CAP(51, 1) /* DerivativeControl */                                                                                 \
    CAP(52, e->features.sampleRateShading)                                                                             \
    CAP(55, e->features.shaderStorageImageReadWithoutFormat)                                                           \
    CAP(56, e->features.shaderStorageImageWriteWithoutFormat)                                                          \
    CAP(57, e->features.multiViewport)                                                                                 \
    CAP(61, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0)                                               \
    CAP(62, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_VOTE_BIT) != 0)                                                \
    CAP(63, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0)                                          \
    CAP(64, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0)                                              \
    CAP(65, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0)                                             \
    CAP(66, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT) != 0)                                    \
    CAP(67, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0)                                           \
    CAP(68, (OOT2_VK_SUBGROUP_OPS & VK_SUBGROUP_FEATURE_QUAD_BIT) != 0)                                                \
    CAP(69, e->f12.shaderOutputLayer)                                                                                  \
    CAP(70, e->f12.shaderOutputViewportIndex)                                                                          \
    CAP(71, 1) /* UniformDecoration */                                                                                 \
    CAP(4427, e->f11.shaderDrawParameters)                                                                             \
    CAP(4433, e->f11.storageBuffer16BitAccess)                                                                         \
    CAP(4434, e->f11.uniformAndStorageBuffer16BitAccess)                                                               \
    CAP(4435, e->f11.storagePushConstant16)                                                                            \
    CAP(4436, e->f11.storageInputOutput16)                                                                             \
    CAP(4437, 1) /* DeviceGroup */                                                                                     \
    CAP(4439, e->f11.multiview)                                                                                        \
    CAP(4441, e->f11.variablePointersStorageBuffer)                                                                    \
    CAP(4442, e->f11.variablePointers)                                                                                 \
    CAP(4448, e->f12.storageBuffer8BitAccess)                                                                          \
    CAP(4449, e->f12.uniformAndStorageBuffer8BitAccess)                                                                \
    CAP(4450, e->f12.storagePushConstant8)                                                                             \
    CAP(5283, e->mesh.meshShader && e->mesh.taskShader)                                                                \
    CAP(5301, 1) /* ShaderNonUniform: Vulkan 1.2 */                                                                    \
    CAP(5302, e->f12.runtimeDescriptorArray)                                                                           \
    CAP(5306, e->f12.shaderUniformBufferArrayNonUniformIndexing)                                                       \
    CAP(5307, e->f12.shaderSampledImageArrayNonUniformIndexing)                                                        \
    CAP(5308, e->f12.shaderStorageBufferArrayNonUniformIndexing)                                                       \
    CAP(5309, e->f12.shaderStorageImageArrayNonUniformIndexing)                                                        \
    CAP(5310, e->f12.shaderInputAttachmentArrayNonUniformIndexing)                                                     \
    CAP(5311, e->f12.shaderUniformTexelBufferArrayNonUniformIndexing)                                                  \
    CAP(5312, e->f12.shaderStorageTexelBufferArrayNonUniformIndexing)                                                  \
    CAP(5345, e->f12.vulkanMemoryModel)                                                                                \
    CAP(5346, e->f12.vulkanMemoryModelDeviceScope)                                                                     \
    CAP(5347, e->f12.bufferDeviceAddress)                                                                              \
    CAP(5379, e->f13.shaderDemoteToHelperInvocation)                                                                   \
    CAP(6016, e->f13.shaderIntegerDotProduct)                                                                          \
    CAP(6017, e->f13.shaderIntegerDotProduct)                                                                          \
    CAP(6018, e->f13.shaderIntegerDotProduct)                                                                          \
    CAP(6019, e->f13.shaderIntegerDotProduct)                                                                          \
    CAP(6022, e->coop.cooperativeMatrix)                                                                               \
    CAP(6033, e->atomic_float.shaderBufferFloat32AtomicAdd)

// EXT(name, expression over `e`): the extension names of a module's OpExtension.
#define VKC_EXTS(EXT)                                                                                                  \
    EXT("SPV_KHR_storage_buffer_storage_class", 1)                                                                     \
    EXT("SPV_KHR_variable_pointers", e->f11.variablePointers || e->f11.variablePointersStorageBuffer)                  \
    EXT("SPV_KHR_shader_draw_parameters", e->f11.shaderDrawParameters)                                                 \
    EXT("SPV_KHR_16bit_storage", e->f11.storageBuffer16BitAccess)                                                      \
    EXT("SPV_KHR_8bit_storage", e->f12.storageBuffer8BitAccess)                                                        \
    EXT("SPV_KHR_multiview", e->f11.multiview)                                                                         \
    EXT("SPV_KHR_device_group", 1)                                                                                     \
    EXT("SPV_KHR_float_controls", 1)                                                                                   \
    EXT("SPV_KHR_non_semantic_info", 1)                                                                                \
    EXT("SPV_EXT_descriptor_indexing", e->f12.runtimeDescriptorArray)                                                  \
    EXT("SPV_KHR_vulkan_memory_model", e->f12.vulkanMemoryModel)                                                       \
    EXT("SPV_KHR_physical_storage_buffer", e->f12.bufferDeviceAddress)                                                 \
    EXT("SPV_EXT_demote_to_helper_invocation", e->f13.shaderDemoteToHelperInvocation)                                  \
    EXT("SPV_EXT_mesh_shader", e->mesh.meshShader)                                                                     \
    EXT("SPV_EXT_shader_atomic_float_add", e->atomic_float.shaderBufferFloat32AtomicAdd)                               \
    EXT("SPV_KHR_cooperative_matrix", e->coop.cooperativeMatrix)

void vkc_list(const Oot2VkFacts *f, VkcList *out) {
    memset(out, 0, sizeof *out);
    Oot2VkFacts en;
    oot2_vk_enable(f, &en);
    const Oot2VkFacts *e = &en;
#define VKC_ADD_CAP(v, expr)                                                                                           \
    if (expr) out->caps[out->ncaps++] = (v);
    VKC_CAPS(VKC_ADD_CAP)
#undef VKC_ADD_CAP
#define VKC_ADD_EXT(name, expr)                                                                                        \
    if (expr) out->exts[out->nexts++] = (name);
    VKC_EXTS(VKC_ADD_EXT)
#undef VKC_ADD_EXT
}

int vkc_has_cap(const VkcList *l, uint32_t cap) {
    for (size_t i = 0; i < l->ncaps; i++)
        if (l->caps[i] == cap) return 1;
    return 0;
}

int vkc_check(const VkcList *l, const SpvDecl *d, char *why, size_t cap) {
    for (uint32_t i = 0; i < d->ncaps; i++) {
        if (vkc_has_cap(l, d->caps[i])) continue;
        const char *name = spv_capability_name(d->caps[i]);
        if (why && cap) {
            if (name) snprintf(why, cap, "capability %s (%u) is not in the device's required list", name, d->caps[i]);
            else snprintf(why, cap, "capability %u is not in the device's required list", d->caps[i]);
        }
        return 1;
    }
    for (uint32_t i = 0; i < d->nexts; i++) {
        size_t k = 0;
        while (k < l->nexts && strcmp(l->exts[k], d->exts[i])) k++;
        if (k < l->nexts) continue;
        if (why && cap) snprintf(why, cap, "extension %s is not in the device's required list", d->exts[i]);
        return 1;
    }
    return 0;
}
