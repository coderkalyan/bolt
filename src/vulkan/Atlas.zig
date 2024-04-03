const std = @import("std");
const Instance = @import("Instance.zig");
const App = @import("../App.zig");
const GlyphCache = @import("../GlyphCache.zig");
const Allocator = std.mem.Allocator;

const c = Instance.c;
const Atlas = @This();

const STAGING_CAPACITY: usize = 256;

gpa: Allocator,
vulkan: *const Instance,
gc: *GlyphCache,
mono_width: u32,
mono_height: u32,

ds: c.VkDescriptorSet,
staging: c.VkBuffer,
staging_mem: c.VkDeviceMemory,
staging_mmap: []u8,
atlas: c.VkBuffer,
atlas_mem: c.VkDeviceMemory,
size: u32,
capacity: u32,
table: std.AutoHashMap(u32, Entry),
queue: std.ArrayList(QueueSlot),
bitmaps: std.ArrayList(u8),
glyph_counter: u32,

const Entry = struct {
    glyph_index: u32,
};

const QueueSlot = struct {
    bitmap: struct { start: u32, end: u32 },
    glyph_index: u32,
};

pub fn init(gpa: Allocator, vulkan: *Instance) !Atlas {
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
    // if (c.vkAllocateDescriptorSets(vulkan.device, &ds_atlas_info, &ds_atlas) != c.VK_SUCCESS) {
    const ret = c.vkAllocateDescriptorSets(vulkan.device, &ds_atlas_info, &ds_atlas);
    if (ret != c.VK_SUCCESS) {
        std.debug.print("ret: {}\n", .{ret});
        return error.VkAllocateDescriptorSetsFailed;
    }
    errdefer _ = c.vkFreeDescriptorSets(vulkan.device, vulkan.ds_pool, 1, &ds_atlas);

    const capacity = STAGING_CAPACITY;
    const mono_width = gc.mono_width;
    const mono_height = gc.mono_height;

    var staging: c.VkBuffer = undefined;
    var staging_mem: c.VkDeviceMemory = undefined;
    const staging_size = capacity * mono_width * mono_height * @sizeOf(u8);
    try vulkan.createBuffer(staging_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &staging, &staging_mem);
    errdefer c.vkDestroyBuffer(vulkan.device, staging, null);
    errdefer c.vkFreeMemory(vulkan.device, staging_mem, null);

    var staging_mmap: []u8 = undefined;
    staging_mmap.len = staging_size;
    _ = c.vkMapMemory(vulkan.device, staging_mem, 0, staging_size, 0, @ptrCast(&staging_mmap.ptr));

    var atlas: c.VkBuffer = undefined;
    var atlas_mem: c.VkDeviceMemory = undefined;
    const atlas_size = capacity * mono_width * mono_height * @sizeOf(u8);
    try vulkan.createBuffer(atlas_size, c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &atlas, &atlas_mem);
    errdefer c.vkDestroyBuffer(vulkan.device, atlas, null);
    errdefer c.vkFreeMemory(vulkan.device, atlas_mem, null);

    var queue = std.ArrayList(QueueSlot).init(gpa);
    errdefer queue.deinit();
    try queue.ensureTotalCapacity(capacity);

    var bitmaps = std.ArrayList(u8).init(gpa);
    errdefer bitmaps.deinit();
    try bitmaps.ensureTotalCapacity(capacity * mono_width * mono_height);

    return .{
        .gpa = gpa,
        .vulkan = vulkan,
        .gc = gc,
        .mono_width = mono_width,
        .mono_height = gc.mono_height,
        .ds = ds_atlas,
        .staging = staging,
        .staging_mem = staging_mem,
        .staging_mmap = staging_mmap,
        .atlas = atlas,
        .atlas_mem = atlas_mem,
        .size = 0,
        .capacity = capacity,
        .table = std.AutoHashMap(u32, Entry).init(gpa),
        .queue = queue,
        .bitmaps = bitmaps,
        .glyph_counter = 0,
    };
}

pub fn deinit(self: *Atlas) void {
    const vulkan = self.vulkan;
    c.vkDestroyBuffer(vulkan.device, self.staging, null);
    c.vkFreeMemory(vulkan.device, self.staging_mem, null);
    c.vkDestroyBuffer(vulkan.device, self.atlas, null);
    c.vkFreeMemory(vulkan.device, self.atlas_mem, null);

    self.table.deinit();
}

// queues a character (unicode codepoint) to be uploaded to the cache
// and returns its glyph index
// if the codepoint already exists, nothing is uploaded
pub fn request(self: *Atlas, cp: u32) !u32 {
    // request glyph bitmap from glyph cache
    const bitmap = try self.gc.request(cp);
    // copy it so we own the memory
    const start: u32 = @intCast(self.bitmaps.items.len);
    try self.bitmaps.appendSlice(bitmap);
    const end: u32 = @intCast(self.bitmaps.items.len);

    const glyph_index = self.glyph_counter;
    self.glyph_counter += 1;

    try self.queue.append(.{
        .bitmap = .{ .start = start, .end = end },
        .glyph_index = glyph_index,
    });

    return glyph_index;
}

// uploads all queued glyphs to the gpu atlas
pub fn commit(self: *Atlas) !void {
    const vulkan = self.vulkan;
    std.debug.assert(self.size + self.queue.items.len <= self.capacity);

    const buffer_info: c.VkCommandBufferAllocateInfo = .{
        .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandPool = vulkan.cmd_pool,
        .commandBufferCount = 1,
    };

    var cmd_buffer: c.VkCommandBuffer = undefined;
    _ = c.vkAllocateCommandBuffers(vulkan.device, &buffer_info, &cmd_buffer);
    defer c.vkFreeCommandBuffers(vulkan.device, vulkan.cmd_pool, 1, &cmd_buffer);

    var writes = std.ArrayList(c.VkWriteDescriptorSet).init(self.gpa);
    defer writes.deinit();
    try writes.ensureTotalCapacity(STAGING_CAPACITY);

    var pos: usize = 0;
    var atlas_pos: usize = self.size * self.mono_width * self.mono_height;
    while (pos < self.queue.items.len) {
        const take = @min(self.queue.items.len - pos, STAGING_CAPACITY);
        // upload the next "take" elements, by first copying into the staging
        // buffer and then into the atlas
        for (self.queue.items[pos .. pos + take], 0..) |req, i| {
            const bitmap = self.bitmaps.items[req.bitmap.start..req.bitmap.end];
            const mmap_pos = i * self.mono_width * self.mono_height;
            @memcpy(self.staging_mmap[mmap_pos .. mmap_pos + bitmap.len], bitmap);

            try writes.append(.{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = self.ds,
                .dstBinding = 0,
                .dstArrayElement = req.glyph_index,
                .descriptorCount = 1, // TODO: what does this do
                .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &c.VkDescriptorBufferInfo{
                    .buffer = self.atlas,
                    .offset = atlas_pos,
                    .range = bitmap.len,
                },
                .pImageInfo = null,
                .pTexelBufferView = null,
            });

            atlas_pos += bitmap.len;
        }

        const begin_info: c.VkCommandBufferBeginInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };
        _ = c.vkBeginCommandBuffer(cmd_buffer, &begin_info);

        const region: c.VkBufferCopy = .{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = self.staging_mmap.len,
        };
        c.vkCmdCopyBuffer(cmd_buffer, self.staging, self.atlas, 1, &region);

        if (c.vkEndCommandBuffer(cmd_buffer) != c.VK_SUCCESS) {
            return error.VkCommandBufferRecordFailed;
        }

        const submit_info: c.VkSubmitInfo = .{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd_buffer,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
            .pWaitDstStageMask = null,
        };
        _ = c.vkQueueSubmit(vulkan.comp_queue, 1, &submit_info, @ptrCast(c.VK_NULL_HANDLE));
        _ = c.vkQueueWaitIdle(vulkan.comp_queue);

        pos += take;
    }

    c.vkUpdateDescriptorSets(vulkan.device, @intCast(writes.items.len), writes.items.ptr, 0, null);
}
