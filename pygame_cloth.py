import pygame
import numpy as np
from Entities import *

class Scene3D:
    def __init__(self, width, height):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        self.ground_height = 0  # Ground level for collision
        pygame.display.set_caption("3D Point Mass Simulation")
        self.clock = pygame.time.Clock()
        
        # Camera settings
        self.camera_pos = np.array([0.0, 0.0, -10.0])
        self.camera_rotation = 0
          # Create a cloth
        self.cloth = Cloth(width=20, length=20, resolution_x=20, resolution_y=20,
                          mass_per_point=0.1, spring_stiffness=200.0, spring_damping=-.1)
        
        # Pin the top edge of the cloth to simulate hanging
        self.cloth.pin_top_edge()
        
        # Position the cloth in the scene (move it up)
        self.cloth.translate(0, 3.0, 0)
        
        # Gravity
        self.gravity = np.array([0.0, 9.81, 0.0])
    
    def handle_input(self):
        """Handle keyboard input for camera movement"""
        keys = pygame.key.get_pressed()
        camera_speed = 0.2
        
        if keys[pygame.K_w]:  # Move forward
            self.camera_pos[2] += camera_speed
        if keys[pygame.K_s]:  # Move backward
            self.camera_pos[2] -= camera_speed
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
        # Apply spring forces first
        for spring in self.cloth.get_all_springs():
            spring.apply_spring_force()
        
        # Update point masses
        for point_mass in self.cloth.get_all_point_masses():
            # Skip pinned points - they don't move
            if hasattr(point_mass, 'pinned') and point_mass.pinned:
                continue
                
            # Apply gravity
            point_mass.apply_force(self.gravity * point_mass.mass)
            
            # Simple ground collision
            if point_mass.position[1] < self.ground_height:
                point_mass.position[1] = self.ground_height
                point_mass.velocity[1] = -point_mass.velocity[1] * 0.8  # Bounce with damping
            
            point_mass.update(dt)
    
    def render(self):
        """Render the 3D scene"""
        self.screen.fill((0, 0, 0))  # Clear screen with black
        
        # Draw coordinate system (optional)
        # self.draw_coordinate_system()
        
        # Draw point masses
        for point_mass in self.cloth.get_all_point_masses():
            x_2d, y_2d, depth = point_mass.project_3d_to_2d(
                self.camera_pos, self.width, self.height
            )
            
            # Only draw if in front of camera and on screen
            if depth > 0 and 0 <= x_2d < self.width and 0 <= y_2d < self.height:
                # Adjust size based on distance
                size = max(1, int(point_mass.radius * 10 / depth))
                pygame.draw.circle(self.screen, point_mass.color, (x_2d, y_2d), size)
        
        pygame.display.flip()
    
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
    
    def run(self):
        """Main game loop"""
        running = True
        dt = 0.016  # ~60 FPS
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        pass  
                        # Reset point mass position
                        # if self.point_masses:
                        #     self.point_masses[0].position = np.array([0.0, 2.0, 0.0])
                        #     self.point_masses[0].velocity.fill(0)
            
            self.handle_input()
            self.update_physics(dt)
            self.render()
            self.clock.tick(60)
        
        pygame.quit()
 
if __name__ == "__main__":
    scene = Scene3D(1920, 1080)
    scene.run()