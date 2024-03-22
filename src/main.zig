const std = @import("std");
const Wayland = @import("Wayland.zig");
// const Vulkan = @import("Vulkan.zig");
const pty = @import("pty.zig");
const GlyphCache = @import("GlyphCache.zig");
const App = @import("App.zig");
const Allocator = std.mem.Allocator;

pub fn main() !void {
    const start = std.time.microTimestamp();

    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    // defer std.debug.assert(general_purpose_allocator.deinit() == .ok);

    // var glyph_cache = try GlyphCache.init();
    // defer glyph_cache.deinit();
    // var glyph_cache: GlyphCache = undefined;

    const glyph_done = std.time.microTimestamp();

    var app = try App.init(general_purpose_allocator.allocator());
    defer app.deinit();
    try app.configure();

    // const wayland_done = std.time.microTimestamp();

    const vulkan_done = std.time.microTimestamp();

    switch (try pty.forkpty()) {
        .parent => |parent| {
            std.debug.print("launched child process: pid = {}, fd = {}\n", .{ parent.pid, parent.fd });

            // const buf = try gpa.alloc(u8, 4096);
            // while (true) {
            //     const bytes_read: isize = @bitCast(std.os.linux.read(parent.fd, buf.ptr, buf.len));
            //     if (bytes_read > 0) {
            //         std.debug.print("{s}", .{buf[0..@intCast(bytes_read)]});
            //     } else if (bytes_read < 0) {
            //         // std.debug.print("{}\n", .{std.os.errno(bytes_read)});
            //     }
            // }
        },
        .child => {
            const arg = ?[*:0]const u8;
            const argv: [:null]const arg = &.{ "/usr/bin/bash", @as(arg, @ptrFromInt(0)) };
            const envp: [:null]const arg = &.{ "TERM=foot", @as(arg, @ptrFromInt(0)) };
            _ = std.os.linux.execve("/usr/bin/bash", argv.ptr, envp.ptr);
        },
    }

    // const lipsum = @embedFile("lipsum.txt");

    const end = std.time.microTimestamp();
    std.debug.print("init time taken: {}\n", .{end - start});
    std.debug.print("glyph time taken: {}\n", .{glyph_done - start});
    // std.debug.print("wayland time taken: {}\n", .{wayland_done - glyph_done});
    std.debug.print("vulkan time taken: {}\n", .{vulkan_done - glyph_done});

    app.running = true;
    while (app.running) {
        if (app.wayland.display.dispatch() != .SUCCESS) break;
        try app.vk_swapchain.drawFrame();
    }
}
