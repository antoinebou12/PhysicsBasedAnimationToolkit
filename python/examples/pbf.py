import pbatoolkit as pbat
import numpy as np
import polyscope as ps
import polyscope.imgui as imgui
import argparse

# Constants
sMaxDensity = 1e4

class FluidFactory:
    def __init__(self, rho0):
        self.rho0 = rho0

    def fill_box(self, aabb, particles):
        dx = (1.0 / self.rho0) ** (1/3)
        x_min, y_min, z_min = aabb['min']
        x_max, y_max, z_max = aabb['max']
        x_vals = np.arange(x_min, x_max + dx, dx)
        y_vals = np.arange(y_min, y_max + dx, dx)
        z_vals = np.arange(z_min, z_max + dx, dx)
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    p = pbat.pbf.Particle()
                    p.x = np.array([x, y, z])
                    p.v = np.array([0.0, 0.0, 0.0])
                    p.key = -1
                    particles.append(p)

    def create_double_dam_break(self):
        particles = []
        self.fill_box({'min': [-2.0, 0.0, -1.0], 'max': [-1.6, 2.0, 1.0]}, particles)
        self.fill_box({'min': [1.6, 0.0, -1.0], 'max': [2.0, 2.0, 1.0]}, particles)
        return particles

    def create_dam_break(self):
        particles = []
        self.fill_box({'min': [-2.0, 0.0, -1.0], 'max': [-1.6, 2.0, 1.0]}, particles)
        return particles

    def create_droplet(self):
        particles = []
        center = np.array([0.0, 1.0, 0.0])
        radius = 0.5
        dx = (1.0 / self.rho0) ** (1/3)
        x_vals = np.arange(center[0] - radius, center[0] + radius + dx, dx)
        y_vals = np.arange(center[1] - radius, center[1] + radius + dx, dx)
        z_vals = np.arange(center[2] - radius, center[2] + radius + dx, dx)
        for x in x_vals:
            for y in y_vals:
                for z in z_vals:
                    position = np.array([x, y, z])
                    if np.linalg.norm(position - center) <= radius:
                        p = pbat.pbf.Particle()
                        p.x = position
                        p.v = np.array([0.0, 0.0, 0.0])
                        p.key = -1
                        particles.append(p)
        return particles

class FluidViewer:
    def __init__(self):
        # Initialize simulation parameters
        self.dt = 0.0167
        self.paused = True
        self.step_once = False
        self.show_velocity_vectors = False
        self.color_by_velocity = False
        self.use_cuda = False  # Placeholder, assuming CPU-only implementation
        self.pbf = pbat.pbf.Pbf(radius=0.1, rho0=1000.0, eps=1e-6, maxIter=5, c=0.01, kCorr=0.01)
        self.factory = FluidFactory(self.pbf.rho0)
        self.particles = []
        self.init_fluid(self.factory.create_dam_break())
        self.setup_polyscope()

    def setup_polyscope(self):
        # Setup Polyscope
        ps.set_program_name("PBF Fluid Simulation")
        ps.set_verbosity(0)
        ps.set_use_prefs_file(False)
        ps.set_always_redraw(True)
        ps.set_ssaa_factor(2)
        ps.set_autocenter_structures(True)
        ps.set_autoscale_structures(True)
        ps.set_ground_plane_mode("none")
        ps.init()

        # Register the callback
        ps.set_user_callback(self.draw)

        # Initialize point cloud
        self.update_point_cloud()

    def init_fluid(self, particles):
        self.particles = particles
        self.pbf.setParticles(self.particles)

    def update_point_cloud(self):
        positions = np.array([p.x for p in self.particles])
        if not hasattr(self, 'fluid_points'):
            self.fluid_points = ps.register_point_cloud("Particles", positions)
            self.fluid_points.set_point_radius(0.005)
            self.fluid_points.set_point_render_mode('sphere')
            self.fluid_density = self.fluid_points.add_scalar_quantity("Density", np.zeros(len(self.particles)))
            self.fluid_density.set_enabled(True)
            self.fluid_density.set_map_range((0, sMaxDensity))
        else:
            self.fluid_points.update_point_positions(positions)

    def draw(self):
        # Draw the GUI
        self.draw_gui()

        # Simulation stepping
        if not self.paused or self.step_once:
            self.pbf.step(self.dt)
            self.step_once = False

        # Update visualization
        self.update_fluid()

    def draw_gui(self):
        imgui.begin("Simulation Controls")

        # Simulation controls
        _, self.paused = imgui.checkbox("Pause", self.paused)
        if imgui.button("Step Once"):
            self.step_once = True

        # Integration parameters
        imgui.text("Integration:")
        _, self.dt = imgui.slider_float("Time Step (dt)", self.dt, 0.001, 0.1)

        # Simulation parameters
        imgui.text("Simulation Parameters:")
        _, self.pbf.radius = imgui.slider_float("Kernel Radius (h)", self.pbf.radius, 0.01, 1.0)
        _, self.pbf.maxIter = imgui.slider_int("Density Iterations", self.pbf.maxIter, 1, 100)
        _, self.pbf.rho0 = imgui.slider_float("Rest Density (rho)", self.pbf.rho0, 100.0, sMaxDensity)
        _, self.pbf.c = imgui.slider_float("Artificial Viscosity (c)", self.pbf.c, 0.0, 0.0002)
        _, self.pbf.kCorr = imgui.slider_float("Artificial Pressure Strength (k)", self.pbf.kCorr, 0.0, 1.0)

        # Visualization options
        imgui.text("Visualization:")
        _, self.show_velocity_vectors = imgui.checkbox("Show Velocity Vectors", self.show_velocity_vectors)
        _, self.color_by_velocity = imgui.checkbox("Color by Velocity", self.color_by_velocity)

        # Scenarios
        imgui.text("Scenarios:")
        if imgui.button("Dam Break"):
            self.init_fluid(self.factory.create_dam_break())
            self.update_point_cloud()
        if imgui.button("Double Dam Break"):
            self.init_fluid(self.factory.create_double_dam_break())
            self.update_point_cloud()
        if imgui.button("Droplet"):
            self.init_fluid(self.factory.create_droplet())
            self.update_point_cloud()

        # Particle count
        imgui.text(f"Number of Particles: {len(self.particles)}")

        imgui.end()

    def update_fluid(self):
        # Update particle positions
        positions = np.array([p.x for p in self.particles])
        self.fluid_points.update_point_positions(positions)

        # Update densities
        densities = np.array([p.rho for p in self.particles])
        self.fluid_density.update_scalar_quantity(densities)

        # Update velocities (if needed)
        if self.show_velocity_vectors:
            velocities = np.array([p.v for p in self.particles])
            if not hasattr(self, 'fluid_velocity'):
                self.fluid_velocity = self.fluid_points.add_vector_quantity("Velocity", velocities)
            else:
                self.fluid_velocity.update_vector_quantity(velocities)
        else:
            if hasattr(self, 'fluid_velocity'):
                self.fluid_points.remove_quantity("Velocity")
                delattr(self, 'fluid_velocity')

        # Color by velocity (if enabled)
        if self.color_by_velocity:
            speeds = np.linalg.norm([p.v for p in self.particles], axis=1)
            if not hasattr(self, 'speed_quantity'):
                self.speed_quantity = self.fluid_points.add_scalar_quantity("Speed", speeds)
            else:
                self.speed_quantity.update_scalar_quantity(speeds)
            self.speed_quantity.set_enabled(True)
        else:
            if hasattr(self, 'speed_quantity'):
                self.speed_quantity.set_enabled(False)

if __name__ == "__main__":
    viewer = FluidViewer()
    ps.show()
