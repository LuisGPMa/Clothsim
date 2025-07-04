import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from Entities import *
from enum import Enum



class Scene3D:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        self.ground_height = 0  # Ground level for collision
        self.system_energy = 0.0
        pygame.display.set_caption("3D Point Mass Simulation")
        self.clock = pygame.time.Clock()
        
        # Initialize font for FPS display
        pygame.font.init()
        self.font = pygame.font.Font(None, 36)
        
        # Camera settings
        self.camera_pos = np.array([0.0, 0.0, 100.0])
        self.camera_pitch = 0.0  # Rotation around X axis
        self.camera_yaw = 0.0    # Rotation around Y axis
        self.camera_roll = 0.0   # Rotation around Z axis
        self.paused = True
        # Gravity
        self.gravity = np.array([0.0, -9.81, 0.0])
        
        # Solid objects in the scene
        self.solid_objects = []
        # Cloth objects in the scene
        self.cloths = []
    
    def spawn_cloth(self, cloth: Cloth,position=[0,0,0],pin_top_edge=False, pin_top_corners=False):
        """Spawn a cloth in the scene"""
        self.cloths.append(cloth)
        cloth.translate(position)
        if pin_top_edge:
            cloth.pin_top_edge()
        elif pin_top_corners:
            cloth.pin_top_corners()
        print("Cloth spawned. Num points:", len(cloth.get_all_point_masses()))
        
    def add_solid_object(self, solid_object):
        """Add a solid object to the scene"""
        self.solid_objects.append(solid_object)
        print(f"Added {type(solid_object).__name__} to scene")
    
    def handle_collisions(self):
        """Handle collisions between cloth and solid objects"""
        for cloth in self.cloths:
            for point_mass in cloth.get_all_point_masses():
                # Skip pinned points for collision
                if hasattr(point_mass, 'pinned') and point_mass.pinned:
                    continue

                for solid_object in self.solid_objects:
                    if solid_object.check_collision(point_mass):
                        solid_object.resolve_collision(point_mass)
    
    def play_pause(self):
        """Toggle pause state"""
        self.paused = not self.paused
        if self.paused:
            print("Simulation paused")
        else:
            print("Simulation running")

    def handle_input(self):
        """Handle keyboard input for camera movement"""
        keys = pygame.key.get_pressed()
        camera_speed = 2.5
        camera_rotation_speed = 2.0
        # Camera rotation controls
        if keys[pygame.K_LEFT]:
            self.camera_yaw -= camera_rotation_speed
        if keys[pygame.K_RIGHT]:
            self.camera_yaw += camera_rotation_speed
        if keys[pygame.K_UP]:
            self.camera_pitch -= camera_rotation_speed            
        if keys[pygame.K_DOWN]:            
            self.camera_pitch += camera_rotation_speed
        if keys[pygame.K_w]:  # Move forward
            self.camera_pos[2] -= camera_speed
        if keys[pygame.K_s]:  # Move backward
            self.camera_pos[2] += camera_speed
        if keys[pygame.K_a]:  # Move left
            self.camera_pos[0] -= camera_speed
        if keys[pygame.K_d]:  # Move right
            self.camera_pos[0] += camera_speed
        if keys[pygame.K_q]:  # Move up
            self.camera_pos[1] += camera_speed
        if keys[pygame.K_e]:  # Move down
            self.camera_pos[1] -= camera_speed
            
    def update_physics(self, dt):
        """Update physics for all point masses"""
        # max_velocity = 40.0
        
        # Update solid objects (for moving objects)
        for solid_object in self.solid_objects:
            solid_object.update(dt)
        
        for cloth in self.cloths:
            cloth.timestep(dt)                
        
        # Handle collisions
        self.handle_collisions()
    
    def render_solid_objects(self):
        """Render solid objects in the scene"""
        for obj in self.solid_objects:
            if isinstance(obj, Sphere):
                self.render_sphere(obj)
            elif isinstance(obj, Plane):
                self.render_plane(obj)
            elif isinstance(obj, Box):
                self.render_box(obj)
    
    def render_sphere(self, sphere):
        """Render a sphere using OpenGL"""
        glPushMatrix()
        glTranslatef(*sphere.position)
        glColor3ub(*sphere.color)
        
        # Draw sphere as wireframe
        glBegin(GL_LINE_STRIP)
        for i in range(20):
            for j in range(20):
                theta = 2 * np.pi * i / 20
                phi = np.pi * j / 20
                x = sphere.radius * np.sin(phi) * np.cos(theta)
                y = sphere.radius * np.sin(phi) * np.sin(theta)
                z = sphere.radius * np.cos(phi)
                glVertex3f(x, y, z)
        glEnd()
        
        glPopMatrix()
    
    def render_plane(self, plane):
        """Render a plane as a grid"""
        glColor3ub(*plane.color)
        
        # Create a grid on the plane
        size = 20.0
        grid_lines = 10
        
        # Find two perpendicular vectors to the normal
        if abs(plane.normal[0]) < 0.9:
            tangent1 = np.cross(plane.normal, [1, 0, 0])
        else:
            tangent1 = np.cross(plane.normal, [0, 1, 0])
        tangent1 = tangent1 / np.linalg.norm(tangent1)
        tangent2 = np.cross(plane.normal, tangent1)
        
        glBegin(GL_LINES)
        for i in range(-grid_lines, grid_lines + 1):
            # Lines in tangent1 direction
            start = plane.position + tangent1 * (i * size / grid_lines) - tangent2 * size
            end = plane.position + tangent1 * (i * size / grid_lines) + tangent2 * size
            glVertex3f(*start)
            glVertex3f(*end)
            
            # Lines in tangent2 direction
            start = plane.position + tangent2 * (i * size / grid_lines) - tangent1 * size
            end = plane.position + tangent2 * (i * size / grid_lines) + tangent1 * size
            glVertex3f(*start)
            glVertex3f(*end)
        glEnd()
    
    def render_box(self, box):
        """Render a box as wireframe"""
        glPushMatrix()
        glTranslatef(*box.position)
        glColor3ub(*box.color)
        
        # Box vertices (relative to center)
        half_dims = box.dimensions / 2
        vertices = [
            [-half_dims[0], -half_dims[1], -half_dims[2]],  # 0
            [+half_dims[0], -half_dims[1], -half_dims[2]],  # 1
            [+half_dims[0], +half_dims[1], -half_dims[2]],  # 2
            [-half_dims[0], +half_dims[1], -half_dims[2]],  # 3
            [-half_dims[0], -half_dims[1], +half_dims[2]],  # 4
            [+half_dims[0], -half_dims[1], +half_dims[2]],  # 5
            [+half_dims[0], +half_dims[1], +half_dims[2]],  # 6
            [-half_dims[0], +half_dims[1], +half_dims[2]],  # 7
        ]
        
        # Box edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
        ]
        
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3f(*vertices[edge[0]])
            glVertex3f(*vertices[edge[1]])
        glEnd()
        
        glPopMatrix()
    
    def render(self):
        """Render the 3D scene using OpenGL"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glPointSize(5)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, self.width / self.height, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        # Apply camera rotations: roll (Z), pitch (X), yaw (Y)
        glRotatef(self.camera_roll, 0, 0, 1)
        glRotatef(self.camera_pitch, 1, 0, 0)
        glRotatef(self.camera_yaw, 0, 1, 0)
        glTranslatef(-self.camera_pos[0], -self.camera_pos[1], -self.camera_pos[2])
        
        points = []
        springs = []
        for cloth in self.cloths:
            points.extend(cloth.get_all_point_masses())
            springs.extend(cloth.get_all_springs())
    
        # Draw point masses        
        glBegin(GL_POINTS)
        for point_mass in points:
            glColor3ub(*point_mass.color)
            glVertex3f(*point_mass.position)
        glEnd()

        # Draw springs
        glBegin(GL_LINES)
        for spring in springs:
            if spring.type == SpringType.STRUCTURAL:
                glColor3ub(255, 0, 0)
            elif spring.type == SpringType.SHEAR:
                glColor3ub(0, 255, 0)
            elif spring.type == SpringType.BENDING:
                glColor3ub(0, 0, 255)
            glVertex3f(*spring.point_a.position)
            glVertex3f(*spring.point_b.position)
        glEnd()
        
        # Draw solid objects
        self.render_solid_objects()

        # Draw FPS
        self.draw_fps()

        pygame.display.flip()
    
    def draw_fps(self):
        """Draw frame render time (ms per frame) on screen"""
        # Calculate ms per frame
        ms_per_frame = self.clock.get_time()
        
        # Switch to 2D rendering for text overlay
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.width, self.height, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glDisable(GL_DEPTH_TEST)
        
        # Create and render ms/frame text
        ms_text = f"Frame: {ms_per_frame:.1f} ms"
        text_surface = self.font.render(ms_text, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surface, "RGBA", True)
        glRasterPos2f(10, 30)
        glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        
        # Restore 3D rendering state
        glEnable(GL_DEPTH_TEST)
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_coordinate_system(self):
        """Draw simple coordinate system for reference"""
        origin = np.array([0.0, 0.0, 0.0])
        
        # X-axis (red)
        x_end = np.array([2.0, 0.0, 0.0])
        self.draw_line_3d(origin, x_end, (255, 0, 0))
        
        # Y-axis (green)
        y_end = np.array([0.0, 2.0, 0.0])
        self.draw_line_3d(origin, y_end, (0, 255, 0))
        
        # Z-axis (blue)
        z_end = np.array([0.0, 0.0, 2.0])
        self.draw_line_3d(origin, z_end, (0, 0, 255))
    
    def draw_line_3d(self, start_3d, end_3d, color):
        """Draw a 3D line by projecting endpoints to 2D"""
        # Create temporary point masses for projection
        start_point = PointMass(start_3d[0], start_3d[1], start_3d[2])
        end_point = PointMass(end_3d[0], end_3d[1], end_3d[2])
        
        start_2d = start_point.project_3d_to_2d(self.camera_pos, self.width, self.height)
        end_2d = end_point.project_3d_to_2d(self.camera_pos, self.width, self.height)
        
        if start_2d[2] > 0 and end_2d[2] > 0:  # Both points in front of camera
            pygame.draw.line(self.screen, color, 
                           (start_2d[0], start_2d[1]), 
                           (end_2d[0], end_2d[1]), 2)
    
    def run(self, dt = 1./30):
        """Main game loop"""
        # OpenGL setup
        glEnable(GL_POINT_SMOOTH)
        glEnable(GL_BLEND)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0, 0, 0, 1)
        running = True
        fps = dt**(-1)
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.play_pause()
            self.handle_input()
            if not self.paused:
                self.update_physics(dt)
            self.render()
            self.clock.tick(fps)
        
        pygame.quit()

if __name__ == "__main__":
    scene = Scene3D(1920, 1080)
    cloth = Cloth(width=20, length=20, resolution_x=40, resolution_y=40,
              mass_per_point=0.1, spring_stiffness=200.0, spring_damping=-.1)
    scene.spawn_cloth(cloth, position=np.array([0, 10, 0]))
    scene.run()