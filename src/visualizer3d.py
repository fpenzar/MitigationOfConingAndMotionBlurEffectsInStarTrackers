from OpenGL.GL import *
from OpenGL.GLU import *
import pygame
from pygame.locals import *
import time
import numpy as np
from gyro_simulator import GyroSimulator
from gyro_integrator import GyroIntegrator
from data_io import DataIO
from quaternion import Quaternion
from threading import Event

def request_window_close():
    try:
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    except pygame.error:
        pass


class Visualizer3D:
    """
    Coordinate convention (logical -> OpenGL):
      - X (right)       -> GL +X
      - Y (into screen) -> GL -Z
      - Z (up)          -> GL +Y

    Euler usage (from q.to_euler(degrees=True) -> roll, pitch, yaw):
      - roll  about X (right)       -> glRotatef( +roll, 1, 0, 0 )
      - pitch about Y (into screen) -> glRotatef( -pitch, 0, 0, 1 )  [about -Z in GL]
      - yaw   about Z (up)          -> glRotatef( +yaw, 0, 1, 0 )    [about +Y in GL]
    """

    def __init__(self, window_size=(800, 600), fov_deg=60.0, near=0.1, far=100.0):
        self.window_size = tuple(window_size)
        self.fov_deg = float(fov_deg)
        self.near = float(near)
        self.far = float(far)
        self._window_created = False

        # Orientation state in degrees
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # Pause toggle (press 'p')
        self._paused = False

        # Text / HUD
        self._font = None
        self._hud_margin = 10
        self._hud_bg = (0, 0, 0, 160)
        self._hud_fg = (255, 255, 255, 255)

        # BODY-frame rotation axis (per-step; inferred from consecutive quats)
        self._axis_body_vec = None     # np.ndarray shape (3,) in logical BODY coords
        self._axis_visible = False
        self._axis_len = 2.5
        self._axis_color = (0.1, 0.1, 0.1)  # dark grey
        self._axis_threshold_rad = np.deg2rad(0.25)  # hide if relative angle < 0.25°

        # viewing angle
        self.view_yaw = 0.0    # rotate around up axis
        self.view_pitch = 0.0  # tilt up/down

    # ---------- Public API ----------

    def create_window(self, caption="3D Visualizer (Z up, X right, Y into screen)"):
        """Create window and initialize GL. Call this separately before drawing/playing."""
        if self._window_created:
            return
        pygame.init()
        pygame.font.init()
        self._font = pygame.font.SysFont("Courier", 16, True)

        # RESIZABLE to receive VIDEORESIZE events
        flags = OPENGL | DOUBLEBUF | RESIZABLE
        self._screen = pygame.display.set_mode(self.window_size, flags)
        pygame.display.set_caption(caption)
        self._resize(*self.window_size)
        self._init_gl()
        self._window_created = True
        self.draw_frame()

    def update_orientation_from_quaternion(self, q):
        """
        q must support q.to_euler(degrees=True) -> np.array/list [roll, pitch, yaw]
        """
        r, p, y = q.to_euler(degrees=True)
        self.roll, self.pitch, self.yaw = float(r), float(p), float(y)
        self.draw_frame()

    def execute_live_maneuver(self, sequence, offset=False):
        """
        Play back a sequence of (Quaternion, timestamp) with real-time pacing.
        During playback, the instantaneous BODY-frame rotation axis is inferred
        from consecutive quaternions and drawn only while the rotation is nonzero.
        sequence: list of (Quaternion, timestamp) pairs (sec, absolute or relative).
        """
        if not self._window_created:
            raise RuntimeError("Call create_window() before execute_live_maneuver().")
        if not sequence:
            return

        # Normalize timestamps to start at 0
        t0 = sequence[0][1]
        rel = [(q, float(t) - float(t0)) for (q, t) in sequence]

        start = time.monotonic()
        q_prev = None
        paused_for = 0
        for q, t in rel:
            # Event + pause handling
            if not self._pump_events():
                break  # window closed
            if self._paused:
                pause_start = time.monotonic()
                while self._paused:
                    if not self._pump_events():
                        return
                    time.sleep(0.02)
                paused_for += time.monotonic() - pause_start

            # Sleep until it's time for this sample
            target = start + max(0.0, t) + paused_for
            now = time.monotonic()
            dt = target - now
            if dt > 0:
                time.sleep(dt)

            # Infer BODY-frame axis from consecutive quaternions
            if q_prev is not None:
                axis_body, ang = self._relative_axis_body(q)
                if axis_body is not None and ang >= self._axis_threshold_rad:
                    self._axis_body_vec = axis_body
                    self._axis_visible = True
                else:
                    self._axis_visible = False
            else:
                self._axis_visible = False

            # Update orientation & draw
            self.update_orientation_from_quaternion(q)
            q_prev = q

        # Hide the axis when the sequence ends
        self._axis_visible = False
        self.draw_frame()

    def keep_running(self, end_evt: Event = None, fps=30):
        """
        Keeps the window open until the user quits (Esc or window close).
        Useful after executing a maneuver to keep the final state visible.
        """
        if not self._window_created:
            raise RuntimeError("Call create_window() before keep_running().")

        clock = pygame.time.Clock()
        while True:
            if end_evt is not None and end_evt.is_set():
                try:
                    pygame.event.post(pygame.event.Event(pygame.USEREVENT, {"name": "end"}))
                except pygame.error:
                    pass
                break
            if not self._pump_events():
                break
            self.draw_frame()
            clock.tick(fps)
        self.close()

    def close(self):
        if self._window_created:
            try:
                pygame.display.quit()
            finally:
                pygame.quit()
            self._window_created = False

    # ---------- Internals ----------

    def _resize(self, width, height):
        height = max(1, int(height))
        width = max(1, int(width))
        self.window_size = (width, height)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fov_deg, width / float(height), self.near, self.far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def _init_gl(self):
        glShadeModel(GL_SMOOTH)
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearDepth(1.0)
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LEQUAL)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def _pump_events(self):
        """Returns False if the user closed the window."""
        for event in pygame.event.get():
            if event.type == QUIT:
                self.close()
                return False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    self.close()
                    return False
                if event.key == K_p:
                    self._paused = not self._paused

                step = 10.0  # deg per keypress
                if event.key == K_LEFT:  self.view_yaw  -= step
                if event.key == K_RIGHT: self.view_yaw  += step
                if event.key == K_UP:    self.view_pitch += step
                if event.key == K_DOWN:  self.view_pitch -= step
            if event.type == VIDEORESIZE:
                self._screen = pygame.display.set_mode((event.w, event.h), OPENGL | DOUBLEBUF | RESIZABLE)
                self._resize(event.w, event.h)
        return True

    def _draw_axes(self, length=2.0, width=2.0):
        glLineWidth(width)
        glBegin(GL_LINES)
        # X axis (right) - Red -> GL +X
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(length, 0.0, 0.0)
        # Y axis (into screen) - Blue -> GL -Z
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, -length)
        # Z axis (up) - Green -> GL +Y
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, length, 0.0)
        glEnd()

        # Axis labels near the tips
        self._draw_text_3d((length + 0.06, 0.0, 0.0), "X")    # +X
        self._draw_text_3d((0.0, 0.0, -length - 0.06), "Y")   # logical +Y is GL -Z
        self._draw_text_3d((0.0, length + 0.06, 0.0), "Z")    # GL +Y is logical +Z

    def _draw_box(self, width=2.0, height=0.5, depth=1.2):
        """
        Draw a rectangular box (cuboid) centered at origin.
        width  -> X axis
        height -> Y axis (up)
        depth  -> Z axis (viewer +/-)
        """
        hx, hy, hz = width * 0.5, height * 0.5, depth * 0.5

        glBegin(GL_QUADS)
        # Top (+Y) - Yellow
        glColor3f(1.0, 1.0, 0.0)
        glVertex3f( hx, hy,  hz)
        glVertex3f(-hx, hy,  hz)
        glVertex3f(-hx, hy, -hz)
        glVertex3f( hx, hy, -hz)

        # Bottom (-Y) - Orange
        glColor3f(1.0, 0.5, 0.0)
        glVertex3f( hx, -hy, -hz)
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, -hy,  hz)
        glVertex3f( hx, -hy,  hz)

        # Front (+Z) - Magenta
        glColor3f(1.0, 0.0, 1.0)
        glVertex3f( hx,  hy,  hz)
        glVertex3f( hx, -hy,  hz)
        glVertex3f(-hx, -hy,  hz)
        glVertex3f(-hx,  hy,  hz)

        # Back (-Z) - Blue
        glColor3f(0.0, 0.0, 1.0)
        glVertex3f( hx, -hy, -hz)
        glVertex3f( hx,  hy, -hz)
        glVertex3f(-hx,  hy, -hz)
        glVertex3f(-hx, -hy, -hz)

        # Right (+X) - Red
        glColor3f(1.0, 0.0, 0.0)
        glVertex3f( hx,  hy, -hz)
        glVertex3f( hx,  hy,  hz)
        glVertex3f( hx, -hy,  hz)
        glVertex3f( hx, -hy, -hz)

        # Left (-X) - Green
        glColor3f(0.0, 1.0, 0.0)
        glVertex3f(-hx,  hy,  hz)
        glVertex3f(-hx,  hy, -hz)
        glVertex3f(-hx, -hy, -hz)
        glVertex3f(-hx, -hy,  hz)
        glEnd()

    def _map_logical_vec_to_gl(self, v3):
        """
        Map a logical BODY vector (x, y, z) to GL local coordinates (x, y, z) for drawing.
        Logical mapping: X->GL +X, Y->GL -Z, Z->GL +Y
        """
        x, y, z = float(v3[0]), float(v3[1]), float(v3[2])
        return np.array([x, z, -y], dtype=float)

    def _draw_rotation_axis_body(self):
        """
        Draw the instantaneous rotation axis in the BODY frame (only when visible).
        This is drawn AFTER applying the object's orientation so it sticks to the body.
        """
        if not (self._axis_visible and self._axis_body_vec is not None):
            return

        axis_gl_local = self._map_logical_vec_to_gl(self._axis_body_vec)
        n = np.linalg.norm(axis_gl_local)
        if n == 0:
            return
        axis_gl_local /= n
        L = float(self._axis_len)
        p0 = -L * axis_gl_local * 0
        p1 =  L * axis_gl_local

        # Draw on top for clarity (no depth)
        glLineWidth(3.0)
        r, g, b = self._axis_color
        glColor3f(r, g, b)
        glBegin(GL_LINES)
        glVertex3f(p0[0], p0[1], p0[2])
        glVertex3f(p1[0], p1[1], p1[2])
        glEnd()
        glEnable(GL_DEPTH_TEST)

    def draw_frame(self):
        """Draw one frame using the current roll/pitch/yaw state, plus HUD."""
        if not self._window_created:
            return

        # --- 3D scene ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # Camera placement: move scene away from camera along GL -Z (standard OpenGL)
        glTranslatef(0.0, 0.0, -6.0)

        glRotatef(self.view_yaw,   0.0, 1.0, 0.0)  # around GL +Y (your logical Z-up)
        glRotatef(self.view_pitch, 1.0, 0.0, 0.0) 

        # Apply orientation to the object
        glRotatef(self.yaw, 0.0, 1.0, 0.0)     # yaw about +Y (Z-up world)
        glRotatef(-self.pitch, 0.0, 0.0, 1.0)  # pitch about -Z (logical Y)
        glRotatef(self.roll, 1.0, 0.0, 0.0)    # roll about +X

        # Draw object and body axes
        self._draw_axes(length=2.0, width=2.0)
        self._draw_box(width=2.0, height=0.5, depth=1.2)

        # Draw the BODY-frame rotation axis (only if active this step)
        self._draw_rotation_axis_body()

        # --- 2D HUD overlay ---
        self._draw_hud()

        pygame.display.flip()

    # ---------- Quaternion helpers ----------

    def _qw_qv(self, q):
        """
        Extract scalar (w) and vector (v) parts from your Quaternion object.
        Supports your typical layout q.vec (x,y,z), q.w.
        """
        try:
            w = float(q.w)
            v = np.asarray(q.vec, dtype=float)
            if v.shape != (3,):
                v = v.reshape(3)
            return w, v
        except Exception as e:
            raise TypeError("Quaternion must expose .w (scalar) and .vec (3-vector).") from e

    def _relative_axis_body(self, q_curr):
        w, v = self._qw_qv(q_curr)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            return None, 0.0

        # Angle in [0, pi]; atan2 is robust to w sign
        angle = 2.0 * np.arctan2(v_norm, abs(w))
        axis = v / v_norm
        return axis, angle

    # ---------- Text / HUD helpers ----------

    def _draw_text_3d(self, pos, text):
        """
        Render text at a 3D world position using glRasterPos and a RGBA sprite.
        Note: the text will be clipped if the position is outside the view frustum.
        """
        if not self._font:
            return
        # Render text to a surface with alpha
        surf = self._font.render(text, True, (0, 0, 0, 255), (255, 255, 255, 0))
        text_data = pygame.image.tostring(surf, "RGBA", True)

        glPushAttrib(GL_LIST_BIT | GL_CURRENT_BIT | GL_ENABLE_BIT | GL_TRANSFORM_BIT)
        glPushMatrix()
        try:
            glRasterPos3f(*pos)
            glDrawPixels(surf.get_width(), surf.get_height(),
                         GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        finally:
            glPopMatrix()
            glPopAttrib()

    def _draw_hud(self):
        """Corner HUD with Roll/Pitch/Yaw in screen space (orthographic)."""
        if not self._font:
            return

        w, h = self.window_size

        # Save current projection/modelview
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)   # 2D orthographic

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Disable depth so HUD is always visible
        glDisable(GL_DEPTH_TEST)

        # Compose text
        text = f"Roll: {self.roll:6.2f}°   Pitch: {self.pitch:6.2f}°   Yaw: {self.yaw:6.2f}°"
        surf = self._font.render(text, True, self._hud_fg, self._hud_bg)
        text_data = pygame.image.tostring(surf, "RGBA", True)
        tw, th = surf.get_width(), surf.get_height()

        # Bottom-left corner with margin (change here for other corners)
        x = self._hud_margin
        y = self._hud_margin

        glRasterPos2i(x, y)
        glDrawPixels(tw, th, GL_RGBA, GL_UNSIGNED_BYTE, text_data)

        # Restore state
        glEnable(GL_DEPTH_TEST)
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)


def coning():
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()

    gyro = GyroSimulator(5)

    omega = 0.5
    half_angle_deg = 30.0
    duration = 100
    dt = 0.01
    
    orientations = gyro.coning(omega, half_angle_deg, duration, dt, use_offset=False)

    viz.execute_live_maneuver(orientations)
    viz.keep_running()


def test():
    dataIO = DataIO()
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()

    gyro = GyroSimulator(5)

    t = 3.0
    w = (np.pi / 2.0) / t

    # t = 5.0
    # w = (np.pi / 4.0) / t

    gyro.set_maneuver(np.array([1, 0, 0]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), -w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), -w, t)
    gyro.set_maneuver(np.array([1, 0, 0]), -w, t)
    
    orientations = gyro.execute_ground_truth_attitude_maneuvers()
    
    # angular_velocities = gyro.execute_attitude_maneuvers()
    # dataIO.save_simulated_ground_truth(orientations)
    # dataIO.save_simulated_angular_velocities(angular_velocities)
    # orientations = dataIO.read_simulated_ground_truth(0)

    viz.execute_live_maneuver(orientations)
    viz.keep_running()

def coning2():
    dataIO = DataIO()
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()
    gyro = GyroSimulator(100, noise_std=0.1)
    omega = 2.0
    half_angle_deg = 10.0
    duration = 10
    s = np.sin(0.5*np.deg2rad(half_angle_deg))
    c = np.cos(0.5*np.deg2rad(half_angle_deg))
    init_quat = Quaternion(s, 0, 0, c)
    gyro_integrator = GyroIntegrator(initial_orientation=init_quat)

    angular_velocities = gyro.coning_angular_velocities(omega, half_angle_deg, duration, noise=True)
    # dataIO.save_simulated_angular_velocities(angular_velocities)
    # angular_velocities = dataIO.read_simulated_angular_velocities(0)
    for w, t in angular_velocities:
        gyro_integrator.update_direct_quaternion(w, t)
    
    viz.execute_live_maneuver(gyro_integrator.orientations)
    viz.keep_running()


def test2():
    dataIO = DataIO()
    viz = Visualizer3D(window_size=(640, 480))
    viz.create_window()
    gyro = GyroSimulator(100, noise_std=0.1)
    gyro_integrator = GyroIntegrator()

    t = 3.0
    w = (np.pi / 2.0) / t
    gyro.set_maneuver(np.array([1, 0, 0]), w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), w, t)
    gyro.set_maneuver(np.array([0, 0, 1]), -w, t)
    gyro.set_maneuver(np.array([0, 1, 0]), -w, t)
    gyro.set_maneuver(np.array([1, 0, 0]), -w, t)

    angular_velocities = gyro.execute_attitude_maneuvers(noise=True)
    # dataIO.save_simulated_angular_velocities(angular_velocities)
    # angular_velocities = dataIO.read_simulated_angular_velocities(0)
    for w, t in angular_velocities:
        gyro_integrator.update_direct_quaternion(w, t)
    viz.execute_live_maneuver(gyro_integrator.orientations)
    viz.keep_running()


if __name__ == "__main__":
    # test()
    # test2()
    coning()
    # coning2()
