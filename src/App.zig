const std = @import("std");
const Wayland = @import("Wayland.zig");
const Vulkan = @import("Vulkan.zig");
const GlyphCache = @import("GlyphCache.zig");
const Allocator = std.mem.Allocator;

const App = @This();

gpa: Allocator,
wayland: Wayland,
vulkan: Vulkan,
glyph_cache: GlyphCache,
running: bool,

// mirrored to GPU via push constant
terminal: Terminal,

pub const Terminal = struct {
    // these can be smaller if necessary but we have the space
    // and shrinking them doesn't help much and causes potential problems
    // with glsl (which likes everything to be 32 bits)

    // window extent (in pixels)
    size: struct { width: u32, height: u32 },
    // number of cells
    cells: struct { cols: u32, rows: u32 },
    // cell size (in pixels)
    cell_size: struct { width: u32, height: u32 },
    // offset to first cell corner
    offset: struct { x: u32, y: u32 },
};

pub fn init(gpa: Allocator) !App {
    const app: App = .{
        .gpa = gpa,
        .wayland = try Wayland.init(),
        .glyph_cache = try GlyphCache.init(),
        .vulkan = undefined,
        .running = false,
        .terminal = .{
            .size = .{
                .width = 0,
                .height = 0,
            },
            .cells = .{ .cols = 0, .rows = 0 },
            .cell_size = .{
                .width = 10,
                .height = 25,
            },
            .offset = .{ .x = 0, .y = 0 },
        },
    };

    return app;
}

pub fn configure(app: *App) !void {
    try app.wayland.configureToplevel();
    app.vulkan = try Vulkan.init(app);
    try app.vulkan.initBufferObjects();
}

pub fn deinit(app: *App) void {
    app.vulkan.deinit();
    app.wayland.deinit();
    app.glyph_cache.deinit();
}

pub fn configureTerminal(app: *App, width_hint: u32, height_hint: u32) void {
    const width = if (width_hint > 0) width_hint else 640;
    const height = if (height_hint > 0) height_hint else 360;

    const cell_width = app.terminal.cell_size.width;
    const cell_height = app.terminal.cell_size.height;

    app.terminal = .{
        .size = .{ .width = width, .height = height },
        .cells = .{ .cols = width / cell_width, .rows = height / cell_height },
        .cell_size = .{ .width = cell_width, .height = cell_height },
        .offset = .{ .x = (width % cell_width) / 2, .y = (height % cell_height) / 2 },
    };

    // std.debug.print("{} {} {} {} {} {}\n", .{ app.terminal.size.width, app.terminal.size.height, app.terminal.cell_size.width, app.terminal.cell_size.height, app.terminal.cells.cols, app.terminal.cells.rows });
}
