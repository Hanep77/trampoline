"""
3D Trampoline Physics Simulation  –  Circular Edition
======================================================
Dependencies:
    pip install moderngl moderngl-window numpy pyrr

Controls:
    Mouse drag (left button)  – Orbit camera
    Mouse scroll              – Zoom in / out
    R                         – Reset all
    SPACE                     – Drop ball from centre above
    W / S / A / D             – Give ball horizontal velocity (N/S/W/E)
    Q / E                     – Decrease / Increase throw speed
    Arrow UP / DOWN           – Increase / Decrease ball drop height

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
const vec3 LIGHT1       = normalize(vec3( 1.5,  2.0,  3.0));
const vec3 LIGHT2       = normalize(vec3(-1.0, -1.0,  1.5));
const vec3 FABRIC_BASE  = vec3(0.10, 0.14, 0.30);
const vec3 FABRIC_HIGH  = vec3(0.22, 0.36, 0.72);
const vec3 IMPACT_COL   = vec3(0.95, 0.80, 0.20);
void main() {
    vec3 n = normalize(v_normal);
    if (!gl_FrontFacing) n = -n;
    float d1  = max(dot(n, LIGHT1), 0.0);
    float d2  = max(dot(n, LIGHT2), 0.0) * 0.25;
    vec3  vd  = normalize(u_cam_pos - v_world_pos);
    vec3  hd  = normalize(LIGHT1 + vd);
    float s   = pow(max(dot(n, hd), 0.0), 56.0) * 0.45;
    float t   = clamp(-v_depth * 0.6, 0.0, 1.0);
    vec3  col = mix(FABRIC_HIGH, FABRIC_BASE, t);
    col = mix(col, IMPACT_COL, u_impact * clamp(1.0 - t*0.7, 0.0, 1.0));
    fragColor = vec4(col * (0.18 + 0.82*(d1+d2)) + s, 1.0);
}
"""

WIRE_VERT = """
#version 330 core
in vec3 in_position;
uniform mat4 u_mvp;
out float v_depth;
void main() {
    v_depth = in_position.z;
    vec4 c  = u_mvp * vec4(in_position, 1.0);
    c.z    -= 0.0006 * c.w;
    gl_Position = c;
}
"""
WIRE_FRAG = """
#version 330 core
in float v_depth;
out vec4 fragColor;
void main() {
    float b = clamp(1.0 + v_depth * 0.55, 0.06, 1.0);
    fragColor = vec4(b*0.45, b*0.78, b*1.0, b*0.50);
}
"""

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
const vec3 METAL = vec3(0.70, 0.70, 0.70);
void main() {
    vec3 n   = normalize(v_normal);
    float d  = max(dot(n, LIGHT), 0.0);
    vec3  vd = normalize(u_cam_pos - v_world_pos);
    vec3  hd = normalize(LIGHT + vd);
    float s  = pow(max(dot(n, hd), 0.0), 90.0) * 0.85;
    fragColor = vec4(METAL*(0.12 + 0.88*d) + s, 1.0);
}
"""

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
    vec3 n   = normalize(v_normal);
    float d1 = max(dot(n, LIGHT1), 0.0);
    float d2 = max(dot(n, LIGHT2), 0.0) * 0.35;
    vec3  vd = normalize(u_cam_pos - v_world_pos);
    vec3  hd = normalize(LIGHT1 + vd);
    float s  = pow(max(dot(n, hd), 0.0), 72.0) * 0.65;
    fragColor = vec4(u_ball_color*(0.15 + 0.85*(d1+d2)) + s, 1.0);
}
"""

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
        verts.append((math.cos(a)*radius, math.sin(a)*radius, 0.0))
    return np.array(verts, dtype=np.float32)


def make_cylinder(r, h, segs=16):
    verts, normals, indices = [], [], []
    for i in range(segs):
        a0 = 2*math.pi*i/segs
        a1 = 2*math.pi*(i+1)/segs
        for a in (a0, a1):
            nx, ny = math.cos(a), math.sin(a)
            verts  += [(nx*r, ny*r, 0.0), (nx*r, ny*r, h)]
            normals += [(nx, ny, 0.0),    (nx, ny, 0.0)]
        b = i*4
        indices += [b, b+1, b+2, b+1, b+3, b+2]
    return (np.array(verts,   dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


def make_torus(R, r, segs_major=64, segs_minor=14):
    verts, normals, indices = [], [], []
    for i in range(segs_major):
        for t, theta in enumerate((2*math.pi*i/segs_major,
                                    2*math.pi*(i+1)/segs_major)):
            ct, st = math.cos(theta), math.sin(theta)
            for j in range(segs_minor):
                phi = 2*math.pi*j/segs_minor
                cp, sp = math.cos(phi), math.sin(phi)
                verts.append(((R+r*cp)*ct, (R+r*cp)*st, r*sp))
                normals.append((ct*cp, st*cp, sp))
        base = i*2*segs_minor
        for j in range(segs_minor):
            n0=base+j; n1=base+(j+1)%segs_minor
            n2=base+segs_minor+j; n3=base+segs_minor+(j+1)%segs_minor
            indices += [n0,n2,n1, n1,n2,n3]
    return (np.array(verts,   dtype=np.float32),
            np.array(normals, dtype=np.float32),
            np.array(indices, dtype=np.uint32))


# ============================================================================
#  CIRCULAR GRID BUILDER
#  Generates a square grid but masks nodes outside the trampoline radius.
#  Masked (outside) nodes are pinned to Z=0 and excluded from spring forces.
# ============================================================================

def build_circular_mask(N, spacing, radius_fraction=0.92):
    """
    Returns a boolean array (N,N): True = node is INSIDE the circle (active).
    Boundary of the circle is always pinned (active but Z=0).
    """
    half  = (N-1)*spacing*0.5
    xs    = np.linspace(-half, half, N)
    ys    = np.linspace(-half, half, N)
    X, Y  = np.meshgrid(xs, ys)
    R     = half * radius_fraction
    dist  = np.sqrt(X**2 + Y**2)
    inside = dist <= R
    return inside, X.astype(np.float32), Y.astype(np.float32), R


def build_circular_mesh(N, spacing, radius_fraction=0.92):
    """
    Build triangle & line index lists only for cells where ALL 4 corners
    are inside the circle.  Returns (tri_idx, line_idx, inside_mask).
    """
    inside, X, Y, R = build_circular_mask(N, spacing, radius_fraction)

    tris  = []
    lines = set()

    for row in range(N-1):
        for col in range(N-1):
            tl = row*N+col; tr = tl+1
            bl = tl+N;      br = bl+1
            if inside[row,col] and inside[row,col+1] and \
               inside[row+1,col] and inside[row+1,col+1]:
                tris += [tl, bl, tr,  tr, bl, br]
                for e in [(tl,tr),(tl,bl),(tr,br),(bl,br)]:
                    lines.add((min(e),max(e)))

    tri_arr  = np.array(tris,          dtype=np.uint32)
    line_arr = np.array(list(lines),   dtype=np.uint32).flatten()
    return tri_arr, line_arr, inside, X, Y, R


def grid_normals_vec(X, Y, Z, spacing):
    """Vectorised per-vertex normals via central finite differences."""
    dx = spacing
    nx = np.zeros_like(Z); ny = np.zeros_like(Z); nz = np.ones_like(Z)
    nx[1:-1,1:-1] = -(Z[1:-1,2:] - Z[1:-1,:-2]) / (2*dx)
    ny[1:-1,1:-1] = -(Z[2:,1:-1] - Z[:-2,1:-1]) / (2*dx)
    L = np.maximum(np.sqrt(nx**2+ny**2+nz**2), 1e-8)
    return nx/L, ny/L, nz/L


# ============================================================================
#  MAIN SIMULATION CLASS
# ============================================================================

class Trampoline(mglw.WindowConfig):
    title        = "3D Trampoline – Circular Physics"
    window_size  = (1280, 720)
    aspect_ratio = 1280/720
    resizable    = True
    gl_version   = (3, 3)
    vsync        = True

    # ── Physics constants ────────────────────────────────────────────────────
    GRID_N          = 52          # internal grid resolution
    GRID_SPACING    = 0.110       # metres between nodes
    # Spring network (8-neighbour)
    K_AXIAL         = 350.0       # horizontal / vertical springs
    K_DIAG          = 220.0       # diagonal springs (×1/√2 rest length)
    K_FLEX          = 80.0        # bending resistance (2-hop neighbour)
    DAMPING         = 2.5         # global velocity damping
    # Ball
    BALL_RADIUS     = 0.30
    BALL_MASS       = 1.0
    GRAVITY         = -9.81
    # Collision
    PENALTY_K       = 2800.0
    RESTITUTION     = 0.78        # energy kept per bounce (vertical)
    FRICTION        = 0.88        # horizontal velocity kept on contact

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        N  = self.GRID_N
        sp = self.GRID_SPACING
        ctx = self.ctx

        # ── Build circular grid ───────────────────────────────────────────────
        (self.tri_idx, self.line_idx,
         self.inside,  self.X, self.Y,
         self.tramp_R) = build_circular_mesh(N, sp, 0.91)

        # Pin mask: nodes ON or OUTSIDE the rim are pinned
        _, _, _, R = build_circular_mask(N, sp, 0.91)
        half = (N-1)*sp*0.5
        dist = np.sqrt(self.X**2 + self.Y**2)
        # Pinned = outside OR on the boundary ring
        self.pinned = ~self.inside  # also pins a thin rim

        self.half_world = half

        # Physics state
        self.Z  = np.zeros((N,N), dtype=np.float32)
        self.Vz = np.zeros((N,N), dtype=np.float32)

        # Ball state
        self.ball_pos    = np.array([0.0, 0.0, 4.0], dtype=np.float32)
        self.ball_vel    = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.throw_speed = 3.0
        self.drop_height = 4.0
        self.impact_intensity = 0.0
        self.in_contact  = False

        # Pre-compute laplacian weights for 8-neighbour spring + bending
        # (done each step via slicing – no explicit pre-build needed)

        # ── Camera ────────────────────────────────────────────────────────────
        self.cam_yaw   =  40.0
        self.cam_pitch =  28.0
        self.cam_dist  =   7.0
        self.mouse_down = False
        self._update_camera()
        self.proj = Matrix44.perspective_projection(
            45.0, self.aspect_ratio, 0.1, 200.0)

        # ── Shaders ───────────────────────────────────────────────────────────
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

        # ── Grid GPU buffers ──────────────────────────────────────────────────
        pos, nor = self._get_grid_data()
        interleaved = self._interleave(pos, nor)
        self.fabric_vbo = ctx.buffer(interleaved.tobytes(), dynamic=True)
        self.fabric_ibo = ctx.buffer(self.tri_idx.tobytes())
        self.fabric_vao = ctx.vertex_array(
            self.fabric_prog,
            [(self.fabric_vbo, "3f 3f", "in_position", "in_normal")],
            self.fabric_ibo)

        self.wire_vbo = ctx.buffer(pos.tobytes(), dynamic=True)
        self.wire_ibo = ctx.buffer(self.line_idx.tobytes())
        self.wire_vao = ctx.vertex_array(
            self.wire_prog,
            [(self.wire_vbo, "3f", "in_position")],
            self.wire_ibo)

        # ── Metal frame ───────────────────────────────────────────────────────
        self._build_frame(ctx)

        # ── Ball ──────────────────────────────────────────────────────────────
        self.ball_vao = sphere(radius=self.BALL_RADIUS).instance(self.ball_prog)

        # ── Shadow disc ───────────────────────────────────────────────────────
        sdv = make_circle_fan(1.0, 40)
        self.shadow_vbo = ctx.buffer(sdv.tobytes())
        self.shadow_vao = ctx.simple_vertex_array(
            self.shadow_prog, self.shadow_vbo, "in_position")

        # ── GL state ──────────────────────────────────────────────────────────
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        print("=" * 60)
        print("  3D TRAMPOLINE  –  Controls")
        print("=" * 60)
        print("  Mouse drag              – Orbit camera")
        print("  Mouse scroll            – Zoom")
        print("  R                       – Full reset")
        print("  SPACE                   – Re-drop ball from centre")
        print("  W / S / A / D           – Throw ball N/S/W/E")
        print("  Q / E                   – Decrease / Increase throw speed")
        print("  Arrow UP / DOWN         – Raise / Lower drop height")
        print("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    #  HELPERS
    # ─────────────────────────────────────────────────────────────────────────

    def _update_camera(self):
        yaw   = math.radians(self.cam_yaw)
        pitch = math.radians(self.cam_pitch)
        cx = self.cam_dist*math.cos(pitch)*math.cos(yaw)
        cy = self.cam_dist*math.cos(pitch)*math.sin(yaw)
        cz = self.cam_dist*math.sin(pitch)
        self.cam_eye = np.array([cx, cy, cz], dtype=np.float32)
        self.view = Matrix44.look_at(
            eye=(cx,cy,cz), target=(0,0,0), up=(0,0,1))

    def _build_frame(self, ctx):
        R = self.tramp_R * 1.04   # ring radius just outside the grid circle
        tv, tn, ti = make_torus(R, 0.045, 72, 14)
        self.ring_vbo = ctx.buffer(np.hstack([tv,tn]).astype("f4").tobytes())
        self.ring_ibo = ctx.buffer(ti.tobytes())
        self.ring_vao = ctx.vertex_array(
            self.frame_prog,
            [(self.ring_vbo, "3f 3f", "in_position", "in_normal")],
            self.ring_ibo)

        self.legs = []
        n_legs = 8
        for i in range(n_legs):
            angle = 2*math.pi*i/n_legs
            cx, cy = R*math.cos(angle), R*math.sin(angle)
            cv, cn, ci = make_cylinder(0.045, 1.5, 14)
            cv[:,0] += cx; cv[:,1] += cy
            cv[:,2]  = -cv[:,2]   # downward
            lv = ctx.buffer(np.hstack([cv,cn]).astype("f4").tobytes())
            li = ctx.buffer(ci.tobytes())
            la = ctx.vertex_array(self.frame_prog,
                                   [(lv,"3f 3f","in_position","in_normal")],li)
            self.legs.append(la)

    def _get_grid_data(self):
        nx, ny, nz = grid_normals_vec(self.X, self.Y, self.Z, self.GRID_SPACING)
        pos = np.stack([self.X.flatten(),
                         self.Y.flatten(),
                         self.Z.flatten()], axis=-1).astype(np.float32)
        nor = np.stack([nx.flatten(),
                         ny.flatten(),
                         nz.flatten()], axis=-1).astype(np.float32)
        return pos, nor

    @staticmethod
    def _interleave(pos, nor):
        return np.hstack([pos, nor]).astype(np.float32)

    def _sample_grid(self, bx, by):
        """Bilinear-interpolate Z, Nx, Ny, Nz at world position (bx,by).
           Returns (gz, nx, ny, nz) – the surface height and unit normal."""
        N  = self.GRID_N
        sp = self.GRID_SPACING
        half = self.half_world

        col_f = (bx + half) / sp
        row_f = (by + half) / sp
        cl = int(np.clip(math.floor(col_f), 0, N-2))
        rl = int(np.clip(math.floor(row_f), 0, N-2))
        fc = max(0.0, min(col_f - cl, 1.0))
        fr = max(0.0, min(row_f - rl, 1.0))

        def bilerp(A):
            return (A[rl,  cl  ]*(1-fr)*(1-fc) +
                    A[rl,  cl+1]*(1-fr)*fc      +
                    A[rl+1,cl  ]*fr*(1-fc)      +
                    A[rl+1,cl+1]*fr*fc)

        gz = bilerp(self.Z)
        nxv, nyv, nzv = grid_normals_vec(
            self.X, self.Y, self.Z, self.GRID_SPACING)
        n = np.array([bilerp(nxv), bilerp(nyv), bilerp(nzv)], dtype=np.float32)
        L = max(np.linalg.norm(n), 1e-8)
        return gz, n/L, int(np.clip(round(col_f),1,N-2)), int(np.clip(round(row_f),1,N-2))

    # ─────────────────────────────────────────────────────────────────────────
    #  PHYSICS  (8-neighbour spring + bending + circular pin)
    # ─────────────────────────────────────────────────────────────────────────

    def _physics_step(self, dt):
        N  = self.GRID_N
        Z  = self.Z
        Vz = self.Vz
        dt = min(dt, 0.018)

        SUBSTEPS = 4
        sub_dt = dt / SUBSTEPS

        for _ in range(SUBSTEPS):
            # ── 8-neighbour spring forces ─────────────────────────────────────
            # Axial (4 direct neighbours, weight 1.0)
            axial = np.zeros_like(Z)
            axial[1:-1,1:-1] = (
                Z[:-2,1:-1] + Z[2:,1:-1] +
                Z[1:-1,:-2] + Z[1:-1,2:] -
                4.0 * Z[1:-1,1:-1])

            # Diagonal (4 diagonal neighbours, weight 1/sqrt(2) ≈ 0.707)
            diag = np.zeros_like(Z)
            diag[1:-1,1:-1] = (
                Z[:-2,:-2] + Z[:-2,2:] +
                Z[2:,:-2]  + Z[2:,2:]  -
                4.0 * Z[1:-1,1:-1]) * 0.7071

            # Bending (2-hop axial neighbours – resist sharp kinks)
            flex = np.zeros_like(Z)
            flex[2:-2,2:-2] = (
                Z[:-4,2:-2] + Z[4:,2:-2] +
                Z[2:-2,:-4] + Z[2:-2,4:] -
                4.0 * Z[2:-2,2:-2]) * 0.5

            force = (self.K_AXIAL * axial +
                     self.K_DIAG  * diag  +
                     self.K_FLEX  * flex  -
                     self.DAMPING * Vz)

            # Only active (inside circle) nodes move
            force[self.pinned] = 0.0

            Vz += force * sub_dt
            Z  += Vz    * sub_dt

            # Re-pin circular boundary & outside nodes every substep
            Z [self.pinned] = 0.0
            Vz[self.pinned] = 0.0

        # ── Ball gravity ──────────────────────────────────────────────────────
        self.ball_vel[2] += self.GRAVITY * dt

        # ── Collision ─────────────────────────────────────────────────────────
        bx, by, bz = self.ball_pos
        br = self.BALL_RADIUS

        # Only collide when ball is over the trampoline disc
        over_tramp = (bx**2 + by**2) < (self.tramp_R * 1.05)**2

        if over_tramp:
            gz, surf_n, col_i, row_i = self._sample_grid(bx, by)

            # Ball contact point = centre - radius * surface_normal
            contact_z = bz - br * surf_n[2]   # approximate bottom
            penetration = gz - contact_z

            self.in_contact = penetration > 0.0

            if self.in_contact:
                # ── Velocity decomposition along surface normal ───────────────
                v  = self.ball_vel.copy()
                vn = np.dot(v, surf_n)          # normal component (signed)

                if vn < 0.0:
                    # Ball moving INTO surface → reflect normal component
                    v_normal_vec = vn * surf_n
                    v_tang_vec   = v - v_normal_vec

                    # Restitution on normal, friction on tangential
                    new_v = (-self.RESTITUTION * v_normal_vec +
                              self.FRICTION    * v_tang_vec)
                    self.ball_vel[:] = new_v

                # Penalty spring pushes ball out along surface normal
                penalty = self.PENALTY_K * penetration
                self.ball_vel += (penalty / self.BALL_MASS * dt) * surf_n

                # ── Grid depression (Gaussian footprint) ─────────────────────
                strength = (penalty / self.BALL_MASS) * dt * 0.30
                r0=max(row_i-3,0); r1=min(row_i+4,N)
                c0=max(col_i-3,0); c1=min(col_i+4,N)
                for rr in range(r0, r1):
                    for cc in range(c0, c1):
                        if self.pinned[rr,cc]: continue
                        d2 = (rr-row_i)**2 + (cc-col_i)**2
                        Vz[rr,cc] -= strength * math.exp(-d2/4.5)

                self.impact_intensity = min(self.impact_intensity + 0.7, 1.0)
            else:
                self.in_contact = False
        else:
            self.in_contact = False

        self.impact_intensity = max(self.impact_intensity - 3.0*dt, 0.0)

        # ── Integrate ball position ───────────────────────────────────────────
        self.ball_pos += self.ball_vel * dt

        # ── Safety reset ──────────────────────────────────────────────────────
        if (self.ball_pos[2] < -4.0 or self.ball_pos[2] > 25.0 or
                math.sqrt(self.ball_pos[0]**2+self.ball_pos[1]**2) > 8.0):
            self._reset()

    def _reset(self):
        self.ball_pos[:] = [0.0, 0.0, self.drop_height]
        self.ball_vel[:] = [0.0, 0.0, 0.0]
        self.Z[:]  = 0.0
        self.Vz[:] = 0.0
        self.impact_intensity = 0.0
        print(f"[Trampoline] Reset. Drop height: {self.drop_height:.1f} m")

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDER
    # ─────────────────────────────────────────────────────────────────────────

    def _do_render(self, time, frame_time):
        ctx = self.ctx
        self._physics_step(frame_time)

        vp    = self.proj * self.view
        mid   = Matrix44.identity()
        cam_b = np.array(self.cam_eye, dtype="f4").tobytes()
        mvp_b = vp.astype("f4").tobytes()
        mid_b = mid.astype("f4").tobytes()

        ctx.clear(0.055, 0.055, 0.095)

        # Upload grid
        pos, nor = self._get_grid_data()
        self.fabric_vbo.write(self._interleave(pos, nor).tobytes())
        self.wire_vbo.write(pos.tobytes())

        # ── Fabric ────────────────────────────────────────────────────────────
        ctx.disable(moderngl.CULL_FACE)
        self.fabric_prog["u_mvp"].write(mvp_b)
        self.fabric_prog["u_model"].write(mid_b)
        self.fabric_prog["u_cam_pos"].write(cam_b)
        self.fabric_prog["u_impact"].value = float(self.impact_intensity)
        self.fabric_vao.render(moderngl.TRIANGLES)

        # ── Wireframe ─────────────────────────────────────────────────────────
        self.wire_prog["u_mvp"].write(mvp_b)
        self.wire_vao.render(moderngl.LINES)

        # ── Frame ─────────────────────────────────────────────────────────────
        self.frame_prog["u_mvp"].write(mvp_b)
        self.frame_prog["u_model"].write(mid_b)
        self.frame_prog["u_cam_pos"].write(cam_b)
        self.ring_vao.render(moderngl.TRIANGLES)
        for leg in self.legs:
            leg.render(moderngl.TRIANGLES)

        # ── Shadow ────────────────────────────────────────────────────────────
        bx, by = self.ball_pos[0], self.ball_pos[1]
        try:
            gz, _, _, _ = self._sample_grid(bx, by)
        except Exception:
            gz = 0.0
        height_above = max(self.ball_pos[2] - self.BALL_RADIUS - gz, 0.0)
        s_scale = self.BALL_RADIUS * (1.0 + height_above*0.45)
        s_alpha = max(0.0, 0.60 - height_above*0.08)
        s_model = (Matrix44.from_translation([bx, by, gz+0.006]) *
                   Matrix44.from_scale([s_scale, s_scale, 1.0]))
        self.shadow_prog["u_mvp"].write((vp*s_model).astype("f4").tobytes())
        self.shadow_prog["u_alpha"].value = float(s_alpha)
        self.shadow_vao.render(moderngl.TRIANGLE_FAN)

        # ── Ball ──────────────────────────────────────────────────────────────
        b_model = Matrix44.from_translation(self.ball_pos.tolist())
        # Warm red that pulses slightly on impact
        pulse   = 0.5 + 0.5*math.sin(time*1.1)
        imp_glow = self.impact_intensity * 0.3
        b_color = np.array([0.88 + imp_glow,
                             0.07 + pulse*0.04,
                             0.07], dtype="f4")
        self.ball_prog["u_mvp"].write((vp*b_model).astype("f4").tobytes())
        self.ball_prog["u_model"].write(b_model.astype("f4").tobytes())
        self.ball_prog["u_cam_pos"].write(cam_b)
        self.ball_prog["u_ball_color"].write(b_color.tobytes())
        self.ball_vao.render()

    # ─────────────────────────────────────────────────────────────────────────
    #  INPUT
    # ─────────────────────────────────────────────────────────────────────────

    def _do_resize(self, w, h):
        if h == 0: return
        self.ctx.viewport = (0,0,w,h)
        self.proj = Matrix44.perspective_projection(45.0, w/h, 0.1, 200.0)

    def _do_key(self, key, action, mod):
        keys = self.wnd.keys
        if action not in (keys.ACTION_PRESS, keys.ACTION_REPEAT): return
        s = self.throw_speed
        if   key == keys.R:
            self._reset()
        elif key == keys.SPACE:
            self.ball_pos[:] = [0.0, 0.0, self.drop_height]
            self.ball_vel[:] = [0.0, 0.0, 0.0]
        elif key == keys.W:     self.ball_vel[1] += s
        elif key == keys.S:     self.ball_vel[1] -= s
        elif key == keys.A:     self.ball_vel[0] -= s
        elif key == keys.D:     self.ball_vel[0] += s
        elif key == keys.Q:
            self.throw_speed = max(0.5, self.throw_speed-0.5)
            print(f"Throw speed: {self.throw_speed:.1f}")
        elif key == keys.E:
            self.throw_speed = min(14.0, self.throw_speed+0.5)
            print(f"Throw speed: {self.throw_speed:.1f}")
        elif key == keys.UP:
            self.drop_height = min(self.drop_height+0.5, 12.0)
            print(f"Drop height: {self.drop_height:.1f} m")
        elif key == keys.DOWN:
            self.drop_height = max(self.drop_height-0.5, 1.0)
            print(f"Drop height: {self.drop_height:.1f} m")

    def _do_mouse_press(self, x, y, btn):
        if btn == 1: self.mouse_down = True

    def _do_mouse_release(self, x, y, btn):
        if btn == 1: self.mouse_down = False

    def _do_mouse_drag(self, x, y, dx, dy, btns):
        if self.mouse_down or btns:
            self.cam_yaw  -= dx*0.35
            self.cam_pitch = float(np.clip(self.cam_pitch+dy*0.35, 5.0, 85.0))
            self._update_camera()

    def _do_scroll(self, xo, yo):
        self.cam_dist = float(np.clip(self.cam_dist-yo*0.4, 2.0, 22.0))
        self._update_camera()

    # ── Dual-API wrappers ─────────────────────────────────────────────────────
    def render(self,t,ft):               self._do_render(t,ft)
    def on_render(self,t,ft):            self._do_render(t,ft)
    def resize(self,w,h):                self._do_resize(w,h)
    def on_resize(self,w,h):             self._do_resize(w,h)
    def key_event(self,k,a,m):           self._do_key(k,a,m)
    def on_key_event(self,k,a,m):        self._do_key(k,a,m)
    def mouse_press_event(self,x,y,b):   self._do_mouse_press(x,y,b)
    def on_mouse_press_event(self,x,y,b):self._do_mouse_press(x,y,b)
    def mouse_release_event(self,x,y,b): self._do_mouse_release(x,y,b)
    def on_mouse_release_event(self,x,y,b):self._do_mouse_release(x,y,b)
    def mouse_drag_event(self,x,y,dx,dy):  self._do_mouse_drag(x,y,dx,dy,True)
    def on_mouse_drag_event(self,x,y,dx,dy):self._do_mouse_drag(x,y,dx,dy,True)
    def mouse_scroll_event(self,xo,yo):     self._do_scroll(xo,yo)
    def on_mouse_scroll_event(self,xo,yo):  self._do_scroll(xo,yo)


# ============================================================================
if __name__ == "__main__":
    mglw.run_window_config(Trampoline)
