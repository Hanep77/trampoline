"""
3D Trampoline Physics Simulation  –  Enhanced Edition
======================================================
Dependencies:
    pip install moderngl moderngl-window numpy pyrr

Controls:
    Mouse drag (left button)  – Orbit camera
    Mouse scroll              – Zoom in / out
    R                         – Reset
    SPACE                     – Drop ball straight down from above
    W/A/S/D                   – Throw ball in XY direction
    Q / E                     – Decrease / Increase throw speed

Run:
    python trampoline_simulation.py
"""

import math

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window.geometry import sphere
from pyrr import Matrix44

# ============================================================================
#  SHADERS
# ============================================================================

# --------------------------------------------------------------------------
# Trampoline FABRIC (solid mesh rendered as triangles)
# --------------------------------------------------------------------------
FABRIC_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3  v_world_pos;
out vec3  v_normal;
out float v_depth;
void main() {
    vec4 wp     = u_model * vec4(in_position, 1.0);
    v_world_pos = wp.xyz;
    v_normal    = normalize(mat3(u_model) * in_normal);
    v_depth     = in_position.z;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

FABRIC_FRAG = """
#version 330 core
in vec3  v_world_pos;
in vec3  v_normal;
in float v_depth;
out vec4 fragColor;
uniform vec3  u_cam_pos;
uniform float u_impact;
const vec3 LIGHT1 = normalize(vec3( 1.5,  2.0,  3.0));
const vec3 LIGHT2 = normalize(vec3(-1.0, -1.0,  2.0));
const vec3 FABRIC_BASE  = vec3(0.08, 0.13, 0.28);
const vec3 FABRIC_LIGHT = vec3(0.18, 0.30, 0.65);
const vec3 SPRING_FLASH = vec3(0.9,  0.75, 0.2);
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    float d1   = max(dot(n, LIGHT1), 0.0);
    float d2   = max(dot(n, LIGHT2), 0.0) * 0.3;
    vec3  vd   = normalize(u_cam_pos - v_world_pos);
    vec3  hd   = normalize(LIGHT1 + vd);
    float s    = pow(max(dot(n, hd), 0.0), 48.0) * 0.4;
    float t    = clamp(-v_depth * 0.5, 0.0, 1.0);
    vec3  base = mix(FABRIC_LIGHT, FABRIC_BASE, t);
    base = mix(base, SPRING_FLASH, u_impact * 0.55);
    vec3 colour = base * (0.20 + 0.80 * (d1 + d2)) + s;
    fragColor = vec4(colour, 1.0);
}
"""

# --------------------------------------------------------------------------
# Wireframe overlay
# --------------------------------------------------------------------------
WIRE_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
out float v_depth;
void main() {
    v_depth     = in_position.z;
    vec4 clip   = u_mvp * vec4(in_position, 1.0);
    clip.z     -= 0.0005 * clip.w;
    gl_Position = clip;
}
"""
WIRE_FRAG = """
#version 330 core
in float v_depth;
out vec4 fragColor;
void main() {
    float b = clamp(1.0 + v_depth * 0.5, 0.08, 1.0);
    fragColor = vec4(b * 0.55, b * 0.85, b * 1.0, b * 0.55);
}
"""

# --------------------------------------------------------------------------
# Metal frame
# --------------------------------------------------------------------------
FRAME_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_world_pos;
void main() {
    vec4 wp     = u_model * vec4(in_position, 1.0);
    v_world_pos = wp.xyz;
    v_normal    = normalize(mat3(u_model) * in_normal);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""
FRAME_FRAG = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 fragColor;
uniform vec3 u_cam_pos;
const vec3 LIGHT = normalize(vec3(1.5, 2.0, 3.0));
const vec3 METAL = vec3(0.72, 0.72, 0.72);
void main() {
    vec3 n  = normalize(v_normal);
    float d = max(dot(n, LIGHT), 0.0);
    vec3  vd = normalize(u_cam_pos - v_world_pos);
    vec3  hd = normalize(LIGHT + vd);
    float s  = pow(max(dot(n, hd), 0.0), 80.0) * 0.8;
    fragColor = vec4(METAL * (0.15 + 0.85 * d) + s, 1.0);
}
"""

# --------------------------------------------------------------------------
# Ball
# --------------------------------------------------------------------------
BALL_VERT = """
#version 330 core
in vec3 in_position;
in vec3 in_normal;
uniform mat4 u_mvp;
uniform mat4 u_model;
out vec3 v_normal;
out vec3 v_world_pos;
void main() {
    vec4 wp     = u_model * vec4(in_position, 1.0);
    v_world_pos = wp.xyz;
    v_normal    = normalize(mat3(u_model) * in_normal);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""
BALL_FRAG = """
#version 330 core
in vec3 v_normal;
in vec3 v_world_pos;
out vec4 fragColor;
uniform vec3 u_cam_pos;
uniform vec3 u_ball_color;
const vec3 LIGHT1 = normalize(vec3( 1.5,  2.0,  3.0));
const vec3 LIGHT2 = normalize(vec3(-1.0, -0.5,  1.5));
void main() {
    vec3 n  = normalize(v_normal);
    float d1 = max(dot(n, LIGHT1), 0.0);
    float d2 = max(dot(n, LIGHT2), 0.0) * 0.35;
    vec3  vd = normalize(u_cam_pos - v_world_pos);
    vec3  hd = normalize(LIGHT1 + vd);
    float s  = pow(max(dot(n, hd), 0.0), 64.0) * 0.6;
    fragColor = vec4(u_ball_color * (0.18 + 0.82 * (d1+d2)) + s, 1.0);
}
"""

# --------------------------------------------------------------------------
# Blob shadow
# --------------------------------------------------------------------------
SHADOW_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
out float v_r;
void main() {
    v_r = length(in_position.xy);
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""
SHADOW_FRAG = """
#version 330 core
in float v_r;
out vec4 fragColor;
uniform float u_alpha;
void main() {
    float a = u_alpha * (1.0 - smoothstep(0.0, 1.0, v_r));
    fragColor = vec4(0.0, 0.0, 0.0, a);
}
"""

# ============================================================================
#  GEOMETRY HELPERS
# ============================================================================

def make_circle_fan(radius, segments=32):
    verts = [(0.0, 0.0, 0.0)]
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        verts.append((math.cos(a) * radius, math.sin(a) * radius, 0.0))
    return np.array(verts, dtype=np.float32)


def make_cylinder(r, h, segs=16):
    verts, normals, indices = [], [], []
    for i in range(segs):
        a0 = 2 * math.pi * i / segs
        a1 = 2 * math.pi * (i + 1) / segs
        for a in (a0, a1):
            nx, ny = math.cos(a), math.sin(a)
            verts  += [(nx*r, ny*r, 0.0), (nx*r, ny*r, h)]
            normals += [(nx, ny, 0.0),   (nx, ny, 0.0)]
        b = i * 4
        indices += [b, b+1, b+2, b+1, b+3, b+2]
    return (np.array(verts,   dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def make_torus(R, r, segs_major=48, segs_minor=12):
    verts, normals, indices = [], [], []
    for i in range(segs_major):
        for t, theta in enumerate((2*math.pi*i/segs_major,
                                    2*math.pi*(i+1)/segs_major)):
            ct, st = math.cos(theta), math.sin(theta)
            for j in range(segs_minor):
                phi = 2 * math.pi * j / segs_minor
                cp, sp = math.cos(phi), math.sin(phi)
                verts.append(((R + r*cp)*ct, (R + r*cp)*st, r*sp))
                normals.append((ct*cp, st*cp, sp))
        base = i * 2 * segs_minor
        for j in range(segs_minor):
            n0 = base + j;           n1 = base + (j+1) % segs_minor
            n2 = base+segs_minor+j;  n3 = base+segs_minor+(j+1)%segs_minor
            indices += [n0, n2, n1, n1, n2, n3]
    return (np.array(verts,   dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def make_grid_triangles(N):
    tris = []
    for row in range(N - 1):
        for col in range(N - 1):
            tl = row*N + col
            tris += [tl, tl+N, tl+1,  tl+1, tl+N, tl+N+1]
    return np.array(tris, dtype=np.uint32)


def grid_normals(X, Y, Z):
    dx = X[0, 1] - X[0, 0]
    nx = np.zeros_like(Z); ny = np.zeros_like(Z); nz = np.ones_like(Z)
    nx[1:-1,1:-1] = -(Z[1:-1,2:] - Z[1:-1,:-2]) / (2*dx)
    ny[1:-1,1:-1] = -(Z[2:,1:-1] - Z[:-2,1:-1]) / (2*dx)
    L = np.maximum(np.sqrt(nx**2 + ny**2 + nz**2), 1e-8)
    return nx/L, ny/L, nz/L


# ============================================================================
#  MAIN CLASS
# ============================================================================

class Trampoline(mglw.WindowConfig):
    title        = "3D Trampoline Physics Simulation – Enhanced"
    window_size  = (1280, 720)
    aspect_ratio = 1280 / 720
    resizable    = True
    gl_version   = (3, 3)
    vsync        = True

    GRID_N       = 48
    GRID_SPACING = 0.115
    SPRING_K     = 320.0
    DAMPING      = 2.2
    BALL_RADIUS  = 0.32
    BALL_MASS    = 1.0
    GRAVITY      = -9.8
    PENALTY_K    = 2200.0
    RESTITUTION  = 0.82

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        N  = self.GRID_N
        sp = self.GRID_SPACING
        ctx = self.ctx

        # Physics
        self.Z  = np.zeros((N, N), dtype=np.float32)
        self.Vz = np.zeros((N, N), dtype=np.float32)
        half = (N - 1) * sp * 0.5
        xs = np.linspace(-half, half, N, dtype=np.float32)
        ys = np.linspace(-half, half, N, dtype=np.float32)
        self.X, self.Y = np.meshgrid(xs, ys)
        self.half_world = half

        self.ball_pos    = np.array([0.0, 0.0, 4.5], dtype=np.float32)
        self.ball_vel    = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.throw_speed = 3.0
        self.impact_intensity = 0.0

        # Camera
        self.cam_yaw   =  45.0
        self.cam_pitch =  30.0
        self.cam_dist  =   6.5
        self.mouse_down = False
        self._update_camera()

        self.proj = Matrix44.perspective_projection(
            45.0, self.aspect_ratio, 0.1, 200.0)

        # Index buffers (static)
        self.grid_tri_idx  = make_grid_triangles(N)
        self.grid_line_idx = self._build_line_idx()

        # Shaders
        self.fabric_prog = ctx.program(vertex_shader=FABRIC_VERT,
                                        fragment_shader=FABRIC_FRAG)
        self.wire_prog   = ctx.program(vertex_shader=WIRE_VERT,
                                        fragment_shader=WIRE_FRAG)
        self.frame_prog  = ctx.program(vertex_shader=FRAME_VERT,
                                        fragment_shader=FRAME_FRAG)
        self.ball_prog   = ctx.program(vertex_shader=BALL_VERT,
                                        fragment_shader=BALL_FRAG)
        self.shadow_prog = ctx.program(vertex_shader=SHADOW_VERT,
                                        fragment_shader=SHADOW_FRAG)

        # Grid GPU buffers
        pos, nor = self._get_grid_data()
        self.fabric_vbo = ctx.buffer(self._interleave(pos, nor).tobytes(),
                                      dynamic=True)
        self.fabric_ibo = ctx.buffer(self.grid_tri_idx.tobytes())
        self.fabric_vao = ctx.vertex_array(
            self.fabric_prog,
            [(self.fabric_vbo, "3f 3f", "in_position", "in_normal")],
            self.fabric_ibo)

        self.wire_vbo = ctx.buffer(pos.tobytes(), dynamic=True)
        self.wire_ibo = ctx.buffer(self.grid_line_idx.tobytes())
        self.wire_vao = ctx.vertex_array(
            self.wire_prog,
            [(self.wire_vbo, "3f", "in_position")],
            self.wire_ibo)

        # Metal frame
        self._build_frame(ctx, half)

        # Ball
        ball_geo = sphere(radius=self.BALL_RADIUS)
        self.ball_vao = ball_geo.instance(self.ball_prog)

        # Shadow disc
        sdv = make_circle_fan(1.0, 32)
        self.shadow_vbo = ctx.buffer(sdv.tobytes())
        self.shadow_vao = ctx.simple_vertex_array(
            self.shadow_prog, self.shadow_vbo, "in_position")

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        print("=" * 58)
        print("  3D TRAMPOLINE SIMULATION  –  Controls")
        print("=" * 58)
        print("  Mouse drag          – Orbit camera")
        print("  Mouse scroll        – Zoom")
        print("  R                   – Reset all")
        print("  SPACE               – Drop ball from above")
        print("  W/A/S/D             – Throw ball (N/W/S/E)")
        print("  Q / E               – Decrease / Increase throw speed")
        print("=" * 58)

    # ─────────────────────────────────────────────────────────────────────────

    def _update_camera(self):
        yaw   = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        cx = self.cam_dist * math.cos(pitch) * math.cos(yaw)
        cy = self.cam_dist * math.cos(pitch) * math.sin(yaw)
        cz = self.cam_dist * math.sin(pitch)
        self.cam_eye = np.array([cx, cy, cz], dtype=np.float32)
        self.view = Matrix44.look_at(
            eye=(cx, cy, cz), target=(0, 0, 0), up=(0, 0, 1))

    def _build_line_idx(self):
        N = self.GRID_N
        lines = []
        for r in range(N):
            for c in range(N):
                i = r * N + c
                if c + 1 < N: lines.append((i, i + 1))
                if r + 1 < N: lines.append((i, i + N))
        return np.array(lines, dtype=np.uint32).flatten()

    def _get_grid_data(self):
        nx, ny, nz = grid_normals(self.X, self.Y, self.Z)
        pos = np.stack([self.X.flatten(), self.Y.flatten(),
                         self.Z.flatten()], axis=-1).astype(np.float32)
        nor = np.stack([nx.flatten(), ny.flatten(),
                         nz.flatten()], axis=-1).astype(np.float32)
        return pos, nor

    @staticmethod
    def _interleave(pos, nor):
        return np.hstack([pos, nor]).astype(np.float32)

    def _build_frame(self, ctx, half):
        ring_r = half * 1.12
        tv, tn, ti = make_torus(ring_r, 0.04, 64, 12)
        self.ring_vbo = ctx.buffer(np.hstack([tv, tn]).astype("f4").tobytes())
        self.ring_ibo = ctx.buffer(ti.tobytes())
        self.ring_vao = ctx.vertex_array(
            self.frame_prog,
            [(self.ring_vbo, "3f 3f", "in_position", "in_normal")],
            self.ring_ibo)

        self.legs = []
        for i in range(6):
            angle = math.pi / 3 * i
            cx, cy = ring_r * math.cos(angle), ring_r * math.sin(angle)
            cv, cn, ci = make_cylinder(0.04, 1.4, 12)
            cv[:, 0] += cx; cv[:, 1] += cy
            cv[:, 2]  = -cv[:, 2]   # legs go downward
            lv = ctx.buffer(np.hstack([cv, cn]).astype("f4").tobytes())
            li = ctx.buffer(ci.tobytes())
            la = ctx.vertex_array(self.frame_prog,
                                   [(lv, "3f 3f", "in_position", "in_normal")],
                                   li)
            self.legs.append(la)

    # ─────────────────────────────────────────────────────────────────────────
    #  PHYSICS
    # ─────────────────────────────────────────────────────────────────────────

    def _physics_step(self, dt):
        N   = self.GRID_N
        Z   = self.Z
        Vz  = self.Vz
        dt  = min(dt, 0.018)
        SUBSTEPS = 3
        sub_dt = dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            lap = np.zeros_like(Z)
            lap[1:-1,1:-1] = (Z[:-2,1:-1] + Z[2:,1:-1] +
                               Z[1:-1,:-2] + Z[1:-1,2:] -
                               4.0 * Z[1:-1,1:-1])
            Vz += (self.SPRING_K * lap - self.DAMPING * Vz) * sub_dt
            Z  += Vz * sub_dt
            Z[0,:]=Vz[0,:]=0.0; Z[-1,:]=Vz[-1,:]=0.0
            Z[:,0]=Vz[:,0]=0.0; Z[:,-1]=Vz[:,-1]=0.0

        # Ball gravity
        self.ball_vel[2] += self.GRAVITY * dt

        # Collision
        bx, by, bz = self.ball_pos
        br   = self.BALL_RADIUS
        half = self.half_world
        sp   = self.GRID_SPACING

        col_f = (bx + half) / sp
        row_f = (by + half) / sp
        col_i = int(np.clip(round(col_f), 1, N-2))
        row_i = int(np.clip(round(row_f), 1, N-2))

        # Bilinear interpolated grid height
        cl = int(np.clip(math.floor(col_f), 0, N-2))
        rl = int(np.clip(math.floor(row_f), 0, N-2))
        fc = col_f - cl; fr = row_f - rl
        gz = (Z[rl,   cl  ]*(1-fr)*(1-fc) + Z[rl,   cl+1]*(1-fr)*fc +
              Z[rl+1, cl  ]*fr*(1-fc)     + Z[rl+1, cl+1]*fr*fc)

        penetration = gz - (bz - br)
        self.in_contact = penetration > 0.0

        if self.in_contact:
            penalty_force = self.PENALTY_K * penetration
            if self.ball_vel[2] < 0:
                self.ball_vel[2] *= -self.RESTITUTION
            self.ball_vel[2] += (penalty_force / self.BALL_MASS) * dt

            impact_strength = (penalty_force / self.BALL_MASS) * dt * 0.28
            r0=max(row_i-2,1); r1=min(row_i+3,N-1)
            c0=max(col_i-2,1); c1=min(col_i+3,N-1)
            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    d2 = (rr-row_i)**2 + (cc-col_i)**2
                    Vz[rr,cc] -= impact_strength * math.exp(-d2/3.0)

            self.impact_intensity = min(self.impact_intensity + 0.6, 1.0)

        self.impact_intensity = max(self.impact_intensity - 2.5*dt, 0.0)
        self.ball_pos += self.ball_vel * dt

        # Safety reset
        max_xy = half * 1.5
        if (abs(self.ball_pos[0]) > max_xy or abs(self.ball_pos[1]) > max_xy
                or self.ball_pos[2] < -3.0 or self.ball_pos[2] > 20.0):
            self._reset()

    def _reset(self):
        self.ball_pos[:] = [0.0, 0.0, 4.5]
        self.ball_vel[:] = [0.0, 0.0, 0.0]
        self.Z[:] = self.Vz[:] = 0.0
        self.impact_intensity = 0.0
        print("[Trampoline] Reset.")

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER
    # ─────────────────────────────────────────────────────────────────────────

    def _do_render(self, time, frame_time):
        ctx = self.ctx
        self._physics_step(frame_time)

        vp      = self.proj * self.view
        mid     = Matrix44.identity()
        cam_pos = self.cam_eye.tolist()
        mvp_b   = vp.astype("f4").tobytes()
        mid_b   = mid.astype("f4").tobytes()
        cam_b   = np.array(cam_pos, dtype="f4").tobytes()

        ctx.clear(0.06, 0.06, 0.10)

        # Upload grid
        pos, nor = self._get_grid_data()
        self.fabric_vbo.write(self._interleave(pos, nor).tobytes())
        self.wire_vbo.write(pos.tobytes())

        # Fabric
        ctx.disable(moderngl.CULL_FACE)
        self.fabric_prog["u_mvp"].write(mvp_b)
        self.fabric_prog["u_model"].write(mid_b)
        self.fabric_prog["u_cam_pos"].write(cam_b)
        self.fabric_prog["u_impact"].value = float(self.impact_intensity)
        self.fabric_vao.render(moderngl.TRIANGLES)

        # Wire overlay
        self.wire_prog["u_mvp"].write(mvp_b)
        self.wire_vao.render(moderngl.LINES)

        # Frame
        self.frame_prog["u_mvp"].write(mvp_b)
        self.frame_prog["u_model"].write(mid_b)
        self.frame_prog["u_cam_pos"].write(cam_b)
        self.ring_vao.render(moderngl.TRIANGLES)
        for leg in self.legs:
            leg.render(moderngl.TRIANGLES)

        # Shadow
        bx, by  = self.ball_pos[0], self.ball_pos[1]
        N, half, sp = self.GRID_N, self.half_world, self.GRID_SPACING
        cl = int(np.clip(math.floor((bx+half)/sp), 0, N-2))
        rl = int(np.clip(math.floor((by+half)/sp), 0, N-2))
        fc = (bx+half)/sp - cl; fr = (by+half)/sp - rl
        fc = max(0.0, min(fc, 1.0)); fr = max(0.0, min(fr, 1.0))
        gz = (self.Z[rl,cl]*(1-fr)*(1-fc) + self.Z[rl,cl+1]*(1-fr)*fc +
              self.Z[rl+1,cl]*fr*(1-fc)   + self.Z[rl+1,cl+1]*fr*fc)
        height_above = max(self.ball_pos[2] - self.BALL_RADIUS - gz, 0.0)
        s_scale = self.BALL_RADIUS * (1.0 + height_above * 0.4)
        s_alpha = max(0.0, 0.55 - height_above * 0.07)
        s_model = (Matrix44.from_translation([bx, by, gz + 0.005]) *
                   Matrix44.from_scale([s_scale, s_scale, 1.0]))
        self.shadow_prog["u_mvp"].write((vp * s_model).astype("f4").tobytes())
        self.shadow_prog["u_alpha"].value = float(s_alpha)
        self.shadow_vao.render(moderngl.TRIANGLE_FAN)

        # Ball
        b_model = Matrix44.from_translation(self.ball_pos.tolist())
        hs = 0.5 + 0.5 * math.sin(time * 0.7)
        b_color = np.array([0.85 + 0.15*hs, 0.08, 0.08 + 0.10*(1-hs)],
                            dtype="f4")
        self.ball_prog["u_mvp"].write((vp * b_model).astype("f4").tobytes())
        self.ball_prog["u_model"].write(b_model.astype("f4").tobytes())
        self.ball_prog["u_cam_pos"].write(cam_b)
        self.ball_prog["u_ball_color"].write(b_color.tobytes())
        self.ball_vao.render()

    # ─────────────────────────────────────────────────────────────────────────
    #  INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def _do_resize(self, w, h):
        if h == 0: return
        self.ctx.viewport = (0, 0, w, h)
        self.proj = Matrix44.perspective_projection(45.0, w/h, 0.1, 200.0)

    def _do_key(self, key, action, mod):
        keys = self.wnd.keys
        if action not in (keys.ACTION_PRESS, keys.ACTION_REPEAT): return
        s = self.throw_speed
        if   key == keys.R:     self._reset()
        elif key == keys.SPACE:
            self.ball_pos[:] = [0.0, 0.0, 4.5]; self.ball_vel[:] = 0.0
        elif key == keys.W:     self.ball_vel[1] += s
        elif key == keys.S:     self.ball_vel[1] -= s
        elif key == keys.A:     self.ball_vel[0] -= s
        elif key == keys.D:     self.ball_vel[0] += s
        elif key == keys.Q:
            self.throw_speed = max(1.0, self.throw_speed - 0.5)
            print(f"Throw speed: {self.throw_speed:.1f}")
        elif key == keys.E:
            self.throw_speed = min(12.0, self.throw_speed + 0.5)
            print(f"Throw speed: {self.throw_speed:.1f}")

    def _do_mouse_press(self, x, y, btn):
        if btn == 1: self.mouse_down = True

    def _do_mouse_release(self, x, y, btn):
        if btn == 1: self.mouse_down = False

    def _do_mouse_drag(self, x, y, dx, dy, btns):
        if self.mouse_down or btns:
            self.cam_yaw  -= dx * 0.35
            self.cam_pitch = float(np.clip(self.cam_pitch + dy*0.35, 5.0, 85.0))
            self._update_camera()

    def _do_scroll(self, xo, yo):
        self.cam_dist = float(np.clip(self.cam_dist - yo*0.4, 2.5, 20.0))
        self._update_camera()

    # Dual-API wrappers ───────────────────────────────────────────────────────
    def render(self, t, ft):             self._do_render(t, ft)
    def on_render(self, t, ft):          self._do_render(t, ft)
    def resize(self, w, h):              self._do_resize(w, h)
    def on_resize(self, w, h):           self._do_resize(w, h)
    def key_event(self, k, a, m):        self._do_key(k, a, m)
    def on_key_event(self, k, a, m):     self._do_key(k, a, m)
    def mouse_press_event(self,x,y,b):   self._do_mouse_press(x,y,b)
    def on_mouse_press_event(self,x,y,b):self._do_mouse_press(x,y,b)
    def mouse_release_event(self,x,y,b): self._do_mouse_release(x,y,b)
    def on_mouse_release_event(self,x,y,b):self._do_mouse_release(x,y,b)
    def mouse_drag_event(self,x,y,dx,dy):  self._do_mouse_drag(x,y,dx,dy,True)
    def on_mouse_drag_event(self,x,y,dx,dy):self._do_mouse_drag(x,y,dx,dy,True)
    def mouse_scroll_event(self,xo,yo):     self._do_scroll(xo,yo)
    def on_mouse_scroll_event(self,xo,yo):  self._do_scroll(xo,yo)


# ============================================================================
#  ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    mglw.run_window_config(Trampoline)
