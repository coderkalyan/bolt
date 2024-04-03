const std = @import("std");
const Instance = @import("Instance.zig");
const App = @import("../App.zig");
const GlyphCache = @import("../GlyphCache.zig");
const Allocator = std.mem.Allocator;

const c = Instance.c;
const Atlas = @This();

vulkan: *const Instance,
gc: *GlyphCache,
mono_width: u32,
mono_height: u32,

ds: c.VkDescriptorSet,
buffer: c.VkBuffer,
slot_count: usize,
// slots: []usize,

pub fn init(gpa: Allocator, vulkan: *Instance) !Atlas {
    _ = gpa;
    const app = @fieldParentPtr(App, "vk_instance", vulkan);
    const gc = &app.glyph_cache;

    const ds_atlas_info: c.VkDescriptorSetAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = vulkan.ds_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &vulkan.ds_layout_atlas,
    };

    var ds_atlas: c.VkDescriptorSet = undefined;
    if (c.vkAllocateDescriptorSets(vulkan.device, &ds_atlas_info, &ds_atlas) != c.VK_SUCCESS) {
        return error.VkAllocateDescriptorSetsFailed;
    }
    errdefer _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, 1, &ds_atlas);

    // // upload test buffer as SSBO for compute shader
    // const size = @sizeOf(u8) * lipsum.len;
    // const aligned_size = (size + 256 - 1) & ~(@as(usize, 256) - 1); // TODO: necessary?
    //
    // var atlas_ssbo_staging: c.VkBuffer = undefined;
    // var atlas_mem_staging: c.VkDeviceMemory = undefined;
    // try vulkan.createBuffer(
    //     aligned_size,
    //     c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    //     c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    //     &atlas_ssbo_staging,
    //     &atlas_mem_staging,
    // );
    // defer c.vkDestroyBuffer(vulkan.device, atlas_ssbo_staging, null);
    // defer c.vkFreeMemory(vulkan.device, atlas_mem_staging, null);
    //
    // var data: [*]u8 = undefined;
    // _ = c.vkMapMemory(vulkan.device, atlas_mem_staging, 0, lipsum.len, 0, @ptrCast(&data));
    // @memcpy(data, lipsum);
    // c.vkUnmapMemory(vulkan.device, atlas_mem_staging);
    //
    // var atlas_ssbo: c.VkBuffer = undefined;
    // var atlas_mem: c.VkDeviceMemory = undefined;
    // try vulkan.createBuffer(
    //     aligned_size,
    //     c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    //     c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
    //     &atlas_ssbo,
    //     &atlas_mem,
    // );
    // vulkan.copyBuffer(atlas_ssbo_staging, atlas_ssbo, aligned_size);
    //
    // const ds_atlas_write: c.VkWriteDescriptorSet = .{
    //     .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
    //     .pNext = null,
    //     .dstSet = ds_atlas,
    //     .dstBinding = 0,
    //     .dstArrayElement = 0,
    //     .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    //     .descriptorCount = 1,
    //     .pBufferInfo = &c.VkDescriptorBufferInfo{
    //         .buffer = atlas_ssbo,
    //         .offset = 0,
    //         .range = aligned_size,
    //     },
    //     .pImageInfo = null,
    //     .pTexelBufferView = null,
    // };
    //
    // c.vkUpdateDescriptorSets(vulkan.device, 1, &ds_atlas_write, 0, null);
    return .{
        .vulkan = vulkan,
        .gc = gc,
        .ds = ds_atlas,
        .mono_width = gc.mono_width,
        .mono_height = gc.mono_height,
        .buffer = undefined,
        .slot_count = 0,
    };
}

pub fn deinit(atlas: *Atlas) void {
    _ = atlas;
}

// fn createSheet(atlas: *Atlas) !void {
//
// }
