const std = @import("std");
const linux = std.os.linux;

const c = @cImport({
    @cInclude("sys/ioctl.h");
    @cInclude("fcntl.h");
});

const SECONDARY_FNAME_MAX = 20;

// probably only works on linux, BSDs and macOS not tested
pub fn openpty() !struct { primary: i32, secondary: i32 } {
    // open/create primary
    const primary: i32 = primary: {
        // zig master
        // const ret = linux.open("/dev/ptmx", .{ .ACCMODE = .RDWR, .NOCTTY = true }, 0);
        const ret = linux.open("/dev/ptmx", c.O_RDWR | c.O_NOCTTY, 0);
        if (@as(isize, @bitCast(ret)) < 0) return error.PtyOpenFailed;
        break :primary @bitCast(@as(u32, @intCast(ret)));
    };
    errdefer _ = linux.close(primary);

    // unlock secondary
    if (linux.ioctl(primary, c.TIOCSPTLCK, @intFromPtr(&@as(i32, 0))) != 0) {
        return error.PtyOpenFailed;
    }

    // request secondary pts number
    var pts_num: i32 = undefined;
    if (linux.ioctl(primary, c.TIOCGPTN, @intFromPtr(&pts_num)) != 0) {
        return error.PtyOpenFailed;
    }

    // open secondary
    var buf = [_]u8{undefined} ** SECONDARY_FNAME_MAX;
    const secondary_name = try std.fmt.bufPrintZ(&buf, "/dev/pts/{}", .{pts_num});
    // zig master
    // const ret = linux.open(secondary_name, .{ .ACCMODE = .RDRW, .NOCTTY = true }, 0);
    const ret = linux.open(secondary_name, c.O_RDWR | c.O_NOCTTY, 0);
    if (@as(isize, @bitCast(ret)) < 0) return error.PtyOpenFailed;
    const secondary: i32 = @bitCast(@as(u32, @intCast(ret)));

    return .{
        .primary = primary,
        .secondary = secondary,
    };
}

pub fn forkpty() !union(enum) { child, parent: struct { pid: usize, fd: i32 } } {
    const pty_fds = try openpty();
    errdefer _ = linux.close(pty_fds.primary);
    errdefer _ = linux.close(pty_fds.secondary);

    var pipe_fds = [_]i32{undefined} ** 2;
    // if (linux.pipe2(&pipe_fds, .{ .CLOEXEC = true }) != 0) {
    if (linux.pipe2(&pipe_fds, c.O_CLOEXEC) != 0) {
        return error.PipeOpenFailed;
    }

    const pid = linux.fork();
    if (pid == 0) {
        // child process
        _ = linux.close(pty_fds.primary);
        _ = linux.close(pipe_fds[0]);

        // setup login session on secondary
        // _ = linux.setsid();
        _ = linux.syscall0(.setsid);
        if (linux.ioctl(pty_fds.secondary, c.TIOCSCTTY, 0) != 0) {
            return error.LoginFailed;
        }

        // assign stdin/out/err to fd and close fd
        _ = linux.dup2(pty_fds.secondary, 0);
        _ = linux.dup2(pty_fds.secondary, 1);
        _ = linux.dup2(pty_fds.secondary, 2);
        if (pty_fds.secondary > 2) _ = linux.close(pty_fds.secondary);

        _ = linux.close(pipe_fds[1]);

        return .{ .child = {} };
    } else {
        _ = linux.close(pty_fds.secondary);
        _ = linux.close(pipe_fds[1]);
        _ = linux.close(pipe_fds[0]);

        return .{ .parent = .{ .pid = pid, .fd = pty_fds.primary } };
    }
}
