const std = @import("std");
const wayland = @import("wayland");
const wl = wayland.client.wl;
const xdg = wayland.client.xdg;

const orderZ = std.mem.orderZ;
var zero: u32 = 0;

fn seatListener(seat: *wl.Seat, event: wl.Seat.Event, _: *u32) void {
    switch (event) {
        .capabilities => |caps| {
            // TODO: no pointer capability binding for now
            if (caps.capabilities.keyboard) {
                const keyboard = seat.getKeyboard() catch return;
                keyboard.setListener(*u32, keyboardListener, &zero);
            }
        },
        .name => {},
    }
}

fn keyboardListener(keyboard: *wl.Keyboard, event: wl.Keyboard.Event, _: *u32) void {
    _ = keyboard;
    switch (event) {
        .keymap => {},
        .enter => {},
        .leave => {},
        .key => |key| {
            _ = key;
            std.debug.print("key press event\n", .{});
        },
        .modifiers => |modifiers| {
            _ = modifiers;
            std.debug.print("modifier event\n", .{});
        },
        .repeat_info => {},
    }
}

fn wmBaseListener(wm_base: *xdg.WmBase, event: xdg.WmBase.Event, _: *u32) void {
    switch (event) {
        .ping => |ping| {
            wm_base.pong(ping.serial);
        },
    }
}

fn xdgSurfaceListener(
    xdg_surface: *xdg.Surface,
    event: xdg.Surface.Event,
    state: *Wayland,
) void {
    switch (event) {
        .configure => |configure| {
            xdg_surface.ackConfigure(configure.serial);

            if (state.configured) {
                state.surface.commit();
            }

            state.configured = true;
        },
    }
}

fn xdgToplevelListener(
    _: *xdg.Toplevel,
    event: xdg.Toplevel.Event,
    state: *Wayland,
) void {
    switch (event) {
        .configure => |configure| {
            state.size.width = configure.width;
            state.size.height = configure.height;
        },
        .close => {
            state.running = false;
        },
    }
}

fn registryListener(
    registry: *wl.Registry,
    event: wl.Registry.Event,
    state: *Wayland,
) void {
    switch (event) {
        .global => |global| {
            if (orderZ(u8, global.interface, wl.Seat.getInterface().name) == .eq) {
                const seat = registry.bind(global.name, wl.Seat, 1) catch return;
                seat.setListener(*u32, seatListener, &zero);
            } else if (orderZ(u8, global.interface, wl.Compositor.getInterface().name) == .eq) {
                state.compositor = registry.bind(global.name, wl.Compositor, 1) catch return;
            } else if (orderZ(u8, global.interface, xdg.WmBase.getInterface().name) == .eq) {
                state.wm_base = registry.bind(global.name, xdg.WmBase, 1) catch return;
                state.wm_base.setListener(*u32, wmBaseListener, &zero);
            }
        },
        .global_remove => {},
    }
}

pub const Wayland = struct {
    display: *wl.Display,
    registry: *wl.Registry,
    compositor: *wl.Compositor,
    surface: *wl.Surface,

    wm_base: *xdg.WmBase,
    xdg_surface: *xdg.Surface,
    xdg_toplevel: *xdg.Toplevel,

    configured: bool,
    running: bool,
    size: struct {
        width: i32,
        height: i32,
    },

    pub fn init() !Wayland {
        // TODO: is there a safer way to do this without a lot of nullable mess?
        var state: Wayland = undefined;
        state.running = false;
        state.configured = false;
        state.size = .{ .width = 0, .height = 0 };

        state.display = try wl.Display.connect(null);
        state.registry = try state.display.getRegistry();
        state.registry.setListener(*Wayland, registryListener, &state);

        if (state.display.roundtrip() != .SUCCESS) return error.RoundtripFailed;

        // TODO: resource acquisition checks
        // if (resources.compositor == null) return error.AcquisitionFailed;
        // if (resources.wm_base == null) return error.AcquisitionFailed;

        state.surface = try state.compositor.createSurface();
        state.xdg_surface = try state.wm_base.getXdgSurface(state.surface);
        state.xdg_toplevel = try state.xdg_surface.getToplevel();

        state.xdg_surface.setListener(*Wayland, xdgSurfaceListener, &state);
        state.xdg_toplevel.setListener(*Wayland, xdgToplevelListener, &state);
        state.xdg_toplevel.setTitle("bolt");
        state.xdg_toplevel.setAppId("bolt");
        state.surface.commit();
        while (state.display.dispatch() == .SUCCESS and !state.configured) {}

        return state;
    }

    pub fn mainloop(state: *Wayland) void {
        state.running = true;
        while (state.display.dispatch() == .SUCCESS and state.running) {}
    }

    pub fn deinit(state: *Wayland) void {
        state.xdg_toplevel.destroy();
        state.xdg_surface.destroy();
        state.surface.destroy();
    }
};
