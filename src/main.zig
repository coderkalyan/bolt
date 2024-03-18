const std = @import("std");
const Wayland = @import("Wayland.zig");
const Vulkan = @import("Vulkan.zig");
const pty = @import("pty.zig");
const GlyphCache = @import("GlyphCache.zig");
const Allocator = std.mem.Allocator;

// shared state
pub const App = struct {
    gpa: Allocator,
    wayland: *Wayland,
    vulkan: *Vulkan,

    running: bool,
    width: u32,
    height: u32,
};

pub fn main() !void {
    var general_purpose_allocator = std.heap.GeneralPurposeAllocator(.{}){};
    defer std.debug.assert(general_purpose_allocator.deinit() == .ok);

    var glyph_cache = try GlyphCache.init();
    defer glyph_cache.deinit();

    var app: App = .{
        .gpa = general_purpose_allocator.allocator(),
        .wayland = undefined,
        .vulkan = undefined,
        .running = false,
        .width = 0,
        .height = 0,
    };

    var wayland = try Wayland.init(&app);
    defer wayland.deinit();
    app.wayland = &wayland;

    var vulkan = try Vulkan.init(app.gpa, &app, &glyph_cache);
    defer vulkan.deinit();
    app.vulkan = &vulkan;

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

    const lipsum = @embedFile("lipsum.txt");
    const cells = try app.gpa.alloc(Vulkan.Cell, 212 * 43);
    var k: usize = 0;
    var row: u32 = 0;
    while (row < 43) : (row += 1) {
        var col: u32 = 0;
        while (col < 212) : (col += 1) {
            cells[k] = .{
                .location = [2]u32{ col, row },
                .character = lipsum[k],
            };
            k += 1;
        }
    }
    const vertex_buffer = try vulkan.createCellAttributesBuffer(cells);

    app.running = true;
    while (app.running) {
        if (wayland.display.dispatch() != .SUCCESS) break;
        try vulkan.drawFrame(&.{vertex_buffer});
    }
}
