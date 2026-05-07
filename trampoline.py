"""
3D Trampoline Physics Simulation
=================================
Dependencies:
    pip install moderngl moderngl-window numpy

Run:
    python trampoline_simulation.py
"""

import moderngl
import moderngl_window as mglw
import numpy as np
from moderngl_window.geometry import sphere
from pyrr import Matrix44, Vector3

# ---------------------------------------------------------------------------
# SHADER SOURCE CODE
# ---------------------------------------------------------------------------

# --- Grid (Trampoline) Shaders ---
GRID_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;

uniform mat4 u_mvp;

// Pass the z-depth to the fragment shader for color intensity
out float v_depth;

void main() {
    gl_Position = u_mvp * vec4(in_position, 1.0);
    // Normalize depth: z=0 is bright, z=-deep is dark
    v_depth = in_position.z;
}
"""

GRID_FRAGMENT_SHADER = """
#version 330 core

in float v_depth;
out vec4 fragColor;

void main() {
    // Retro terminal neon green: color fades as grid dips down (z < 0)
    // clamp so it never fully disappears
    float intensity = clamp(1.0 + v_depth * 0.35, 0.15, 1.0);
    fragColor = vec4(0.0, intensity, 0.0, 1.0);
}
"""

# --- Ball Shaders ---
BALL_VERTEX_SHADER = """
#version 330 core

in vec3 in_position;
in vec3 in_normal;

uniform mat4 u_mvp;
uniform mat4 u_model;

out vec3 v_normal_world;
out vec3 v_frag_pos;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_frag_pos    = world_pos.xyz;
    // Transform normals to world space (no non-uniform scale, so model matrix is fine)
    v_normal_world = mat3(u_model) * in_normal;
    gl_Position   = u_mvp * vec4(in_position, 1.0);
}
"""

BALL_FRAGMENT_SHADER = """
#version 330 core

in vec3 v_normal_world;
in vec3 v_frag_pos;

out vec4 fragColor;

// Fixed light direction (from upper-left-front)
const vec3 LIGHT_DIR = normalize(vec3(1.0, 2.0, 3.0));
const vec3 BALL_COLOR = vec3(0.9, 0.05, 0.05);   // solid red

void main() {
    vec3 norm     = normalize(v_normal_world);
    float diffuse = max(dot(norm, LIGHT_DIR), 0.0);
    float ambient = 0.25;
    float light   = ambient + diffuse * 0.75;
    fragColor = vec4(BALL_COLOR * light, 1.0);
}
"""


# ---------------------------------------------------------------------------
# MAIN SIMULATION CLASS
# ---------------------------------------------------------------------------


class Trampoline(mglw.WindowConfig):
    """
    3D Trampoline Physics Simulation using a Mass-Spring system.
    Render the deformable grid as neon-green wireframe and a bouncing ball.
    """

    # moderngl_window configuration
    title = "3D Trampoline Physics Simulation"
    window_size = (1280, 720)
    aspect_ratio = 1280 / 720
    resizable = True
    gl_version = (3, 3)
    vsync = True

    # ------------------------------------------------------------------ #
    # Physics constants                                                    #
    # ------------------------------------------------------------------ #
    GRID_N = 40  # Grid is GRID_N x GRID_N vertices
    GRID_SPACING = 0.125  # World-space distance between adjacent nodes
    # Spring / damping
    SPRING_K = 280.0  # Spring stiffness
    DAMPING = 1.8  # Velocity damping coefficient
    # Ball
    BALL_RADIUS = 0.35
    BALL_MASS = 1.0
    GRAVITY = -9.8
    # Penalty-method collision stiffness
    PENALTY_K = 1800.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ----------------------------------------------------------------
        # 1. PHYSICS STATE  (Numpy arrays – fully vectorized)
        # ----------------------------------------------------------------
        N = self.GRID_N

        # Grid node positions: Z only (X/Y are fixed)
        self.Z = np.zeros((N, N), dtype=np.float32)  # displacement
        self.Vz = np.zeros((N, N), dtype=np.float32)  # velocity

        # Pre-compute fixed XY world coordinates for every node
        half = (N - 1) * self.GRID_SPACING * 0.5
        xs = np.linspace(-half, half, N, dtype=np.float32)
        ys = np.linspace(-half, half, N, dtype=np.float32)
        self.X, self.Y = np.meshgrid(xs, ys)  # shape (N, N)

        # Ball state
        self.ball_pos = np.array([0.0, 0.0, 5.0], dtype=np.float32)
        self.ball_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # ----------------------------------------------------------------
        # 2. BUILD GRID GEOMETRY (indices stay constant, positions update)
        # ----------------------------------------------------------------
        self._build_grid_indices()

        # ----------------------------------------------------------------
        # 3. MODERNGL SETUP
        # ----------------------------------------------------------------
        ctx = self.ctx

        # --- Grid program & buffers ---
        self.grid_prog = ctx.program(
            vertex_shader=GRID_VERTEX_SHADER,
            fragment_shader=GRID_FRAGMENT_SHADER,
        )

        # VBO for grid vertex positions – will be updated every frame
        initial_verts = self._get_grid_vertices()
        self.grid_vbo = ctx.buffer(initial_verts.tobytes(), dynamic=True)

        # IBO (index buffer for LINES) – never changes
        self.grid_ibo = ctx.buffer(self.grid_indices.tobytes())

        self.grid_vao = ctx.vertex_array(
            self.grid_prog,
            [(self.grid_vbo, "3f", "in_position")],
            self.grid_ibo,
        )

        # --- Ball program & geometry ---
        self.ball_prog = ctx.program(
            vertex_shader=BALL_VERTEX_SHADER,
            fragment_shader=BALL_FRAGMENT_SHADER,
        )

        # moderngl_window provides a helper sphere; radius=1 (we scale via model matrix)
        self.ball_geo = sphere(radius=self.BALL_RADIUS)
        self.ball_vao = self.ball_geo.instance(self.ball_prog)

        # Render states
        ctx.enable(moderngl.DEPTH_TEST)

        # ----------------------------------------------------------------
        # 4. CAMERA / PROJECTION
        # ----------------------------------------------------------------
        half_world = (N - 1) * self.GRID_SPACING * 0.5
        cam_distance = half_world * 3.5
        self.proj = Matrix44.perspective_projection(
            fovy=45.0,
            aspect=self.aspect_ratio,
            near=0.1,
            far=200.0,
        )
        self.view = Matrix44.look_at(
            eye=(cam_distance * 0.9, -cam_distance * 1.1, cam_distance * 0.85),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        )
        self.vp = self.proj * self.view  # view-projection (grid has no model matrix)

    # ------------------------------------------------------------------ #
    # Helper: build line-list index buffer                                 #
    # ------------------------------------------------------------------ #
    def _build_grid_indices(self):
        """
        Build index pairs for rendering the grid as LINES.
        Each interior horizontal and vertical edge is one line segment.
        """
        N = self.GRID_N
        lines = []

        for row in range(N):
            for col in range(N):
                idx = row * N + col
                # Horizontal edge →
                if col + 1 < N:
                    lines.append((idx, idx + 1))
                # Vertical edge ↑
                if row + 1 < N:
                    lines.append((idx, idx + N))

        self.grid_indices = np.array(lines, dtype=np.uint32).flatten()

    # ------------------------------------------------------------------ #
    # Helper: pack current Z into a flat vertex array [(x,y,z), ...]      #
    # ------------------------------------------------------------------ #
    def _get_grid_vertices(self) -> np.ndarray:
        """Return shape (N*N, 3) float32 array of current vertex positions."""
        verts = np.stack(
            [self.X.flatten(), self.Y.flatten(), self.Z.flatten()], axis=-1
        ).astype(np.float32)
        return verts

    # ------------------------------------------------------------------ #
    # Physics step                                                         #
    # ------------------------------------------------------------------ #
    def _physics_step(self, dt: float):
        """
        Advance the mass-spring grid and ball by one timestep dt.
        All grid operations are vectorized over the (N,N) arrays.
        """
        N = self.GRID_N
        Z = self.Z
        Vz = self.Vz
        dt = min(dt, 0.02)  # cap dt to avoid blow-ups on hiccups

        # ---- 4-neighbor discrete Laplacian (finite differences) ----
        # Interior: Σ(z_neighbor) - 4*z_center
        laplacian = np.zeros_like(Z)
        laplacian[1:-1, 1:-1] = (
            Z[:-2, 1:-1]  # row above
            + Z[2:, 1:-1]  # row below
            + Z[1:-1, :-2]  # col left
            + Z[1:-1, 2:]  # col right
            - 4.0 * Z[1:-1, 1:-1]
        )
        # Boundary nodes are pinned → their velocities will be zeroed anyway

        # ---- Spring force (Hooke's law) + damping ----
        # F = k * laplacian - d * velocity   (per unit mass, mass=1)
        accel_z = self.SPRING_K * laplacian - self.DAMPING * Vz

        # ---- Semi-Implicit Euler: update V first, then Z ----
        Vz += accel_z * dt
        Z += Vz * dt

        # ---- Pin boundary nodes to Z = 0 ----
        Z[0, :] = 0.0
        Vz[0, :] = 0.0
        Z[-1, :] = 0.0
        Vz[-1, :] = 0.0
        Z[:, 0] = 0.0
        Vz[:, 0] = 0.0
        Z[:, -1] = 0.0
        Vz[:, -1] = 0.0

        # ---- Ball: gravity ----
        self.ball_vel[2] += self.GRAVITY * dt

        # ---- Penalty-method collision: ball vs. trampoline grid ----
        bx, by, bz = self.ball_pos
        br = self.BALL_RADIUS

        # Find the nearest grid node to the ball's XY position
        half = (N - 1) * self.GRID_SPACING * 0.5
        col_f = (bx + half) / self.GRID_SPACING
        row_f = (by + half) / self.GRID_SPACING
        col_i = int(np.clip(np.round(col_f), 1, N - 2))
        row_i = int(np.clip(np.round(row_f), 1, N - 2))

        grid_z_at_ball = self.Z[row_i, col_i]

        # Penetration depth: positive means the ball bottom is below the grid
        penetration = grid_z_at_ball - (bz - br)

        if penetration > 0.0:
            penalty_force = self.PENALTY_K * penetration

            # Push ball upward
            self.ball_vel[2] += (penalty_force / self.BALL_MASS) * dt

            # Push grid downward at impact node (action-reaction)
            # Apply the reaction force to a small 3x3 region for smooth look
            r0 = max(row_i - 1, 1)
            r1 = min(row_i + 2, N - 1)
            c0 = max(col_i - 1, 1)
            c1 = min(col_i + 2, N - 1)
            Vz[r0:r1, c0:c1] -= (penalty_force / self.BALL_MASS) * dt * 0.35

        # ---- Update ball position ----
        self.ball_pos += self.ball_vel * dt

        # Safety floor – prevent ball from falling through forever
        if self.ball_pos[2] < -10.0:
            self.ball_pos[2] = 5.0
            self.ball_vel[:] = 0.0

    # ------------------------------------------------------------------ #
    # moderngl_window main loop callbacks                                  #
    # ------------------------------------------------------------------ #
    def _do_render(self, time: float, frame_time: float):
        """Core render logic, called by both render() and on_render()."""
        ctx = self.ctx

        # ---- Physics ----
        self._physics_step(frame_time)

        # ---- Clear ----
        ctx.clear(0.03, 0.03, 0.06)  # very dark navy – retro terminal bg

        # ---- Render trampoline grid ----
        # Upload updated vertex positions to GPU
        new_verts = self._get_grid_vertices()
        self.grid_vbo.write(new_verts.tobytes())

        mvp = self.vp  # grid lives at world origin, no model matrix
        self.grid_prog["u_mvp"].write(mvp.astype("f4").tobytes())
        self.grid_vao.render(moderngl.LINES)

        # ---- Render ball ----
        # Build model matrix: translate to ball position (radius already baked in sphere)
        model = Matrix44.from_translation(self.ball_pos.tolist())
        mvp_ball = self.proj * self.view * model

        self.ball_prog["u_mvp"].write(mvp_ball.astype("f4").tobytes())
        self.ball_prog["u_model"].write(model.astype("f4").tobytes())
        self.ball_vao.render()

    # Support both old API (render) and new API (on_render)
    def render(self, time: float, frame_time: float):
        self._do_render(time, frame_time)

    def on_render(self, time: float, frame_time: float):
        self._do_render(time, frame_time)

    def _do_resize(self, width: int, height: int):
        if height == 0:
            return
        self.ctx.viewport = (0, 0, width, height)
        self.proj = Matrix44.perspective_projection(
            fovy=45.0,
            aspect=width / height,
            near=0.1,
            far=200.0,
        )
        self.vp = self.proj * self.view

    def resize(self, width: int, height: int):
        self._do_resize(width, height)

    def on_resize(self, width: int, height: int):
        self._do_resize(width, height)

    def _do_key_event(self, key, action, modifiers):
        """Press R to reset the ball to the top."""
        keys = self.wnd.keys
        if action == keys.ACTION_PRESS and key == keys.R:
            self.ball_pos[:] = [0.0, 0.0, 5.0]
            self.ball_vel[:] = [0.0, 0.0, 0.0]
            self.Z[:] = 0.0
            self.Vz[:] = 0.0
            print("[Trampoline] Reset.")

    def key_event(self, key, action, modifiers):
        self._do_key_event(key, action, modifiers)

    def on_key_event(self, key, action, modifiers):
        self._do_key_event(key, action, modifiers)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mglw.run_window_config(Trampoline)
