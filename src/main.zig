const std = @import("std");
const wayland = @import("wayland.zig");
const Vulkan = @import("Vulkan.zig");

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    // defer general_purpose_allocator.deinit();
    const gpa = general_purpose_allocator.allocator();

    var display = try wayland.Wayland.init();
    defer display.deinit();

    var vulkan = try Vulkan.init(gpa, &display);
    defer vulkan.deinit();

    // display.mainloop();

    try vulkan.drawFrame();

    while (true) {
        _ = display.display.roundtrip();
    }
    // var vulkan = try Vulkan.init();
    // defer vulkan.deinit();
}
