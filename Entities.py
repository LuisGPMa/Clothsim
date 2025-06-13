import math
import numpy as np

class PointMass:
    def __init__(self, x, y, z, mass=1.0):
        self.position = np.array([x, y, z], dtype=float)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = mass
        self.color = (255, 255, 255)
        self.radius = 5
    
    def apply_force(self, force):
        """Apply force to the point mass (F = ma)"""
        self.acceleration += force / self.mass
    
    def update(self, dt):
        """Update position and velocity using Verlet integration"""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt
        self.acceleration.fill(0)  # Reset acceleration
    
    def project_3d_to_2d(self, camera_pos, screen_width, screen_height, fov=60):
        """Simple perspective projection"""
        # Translate relative to camera
        rel_pos = self.position - camera_pos
        
        # Avoid division by zero
        if rel_pos[2] <= 0.1:
            rel_pos[2] = 0.1
        
        # Perspective projection
        fov_rad = math.radians(fov)
        f = 1 / math.tan(fov_rad / 2)
        
        x_2d = (rel_pos[0] * f / rel_pos[2]) * screen_width / 2 + screen_width / 2
        y_2d = (rel_pos[1] * f / rel_pos[2]) * screen_height / 2 + screen_height / 2
        
        return int(x_2d), int(y_2d), rel_pos[2]
    
class Spring:
    def __init__(self, point_a: PointMass, point_b: PointMass, rest_length: float, stiffness=1.0, damping=-.1):
        self.point_a = point_a
        self.point_b = point_b
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping
    
    def apply_spring_force(self):
        """Calculate and apply spring force to both connected point masses"""
        # Calculate the vector from point_a to point_b
        displacement = self.point_b.position - self.point_a.position
        
        # Calculate current length of the spring
        current_length = np.linalg.norm(displacement)
        
        # Avoid division by zero
        if current_length == 0:
            return
        
        # Calculate the normalized direction vector
        direction = displacement / current_length
        
        # Calculate the spring force magnitude using Hooke's law: F = -k * (x - x0)
        # where x is current length, x0 is rest length, k is stiffness
        force_magnitude = self.stiffness * (current_length - self.rest_length)
        
        # Calculate force vector (pointing from longer to shorter length)
        force_vector = force_magnitude * direction
        relative_velocity = self.point_b.velocity - self.point_a.velocity
        damping_force = self.damping * np.dot(relative_velocity, direction) * direction

        # Total force
        total_force = force_vector + (-damping_force)
        
        # Apply equal and opposite forces to both point masses
        self.point_a.apply_force(total_force)   # Force pulls point_a toward point_b
        self.point_b.apply_force(-total_force)  # Force pulls point_b toward point_a

class Cloth:
    def __init__(self, width, length, resolution_x, resolution_y, mass_per_point=1.0, spring_stiffness=100.0, spring_damping=-.1):
        """
        Create a cloth as a grid of point masses connected by springs
        
        Args:
            width: Physical width of the cloth
            length: Physical length of the cloth
            resolution_x: Number of point masses along width
            resolution_y: Number of point masses along length
            mass_per_point: Mass of each point mass
            spring_stiffness: Stiffness of connecting springs
        """
        self.width = width
        self.length = length
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.mass_per_point = mass_per_point
        self.spring_stiffness = spring_stiffness
        self.spring_damping = spring_damping
        # Grid of point masses
        self.point_masses = []
        self.springs = []
        
        # Calculate spacing between points
        self.dx = width / (resolution_x - 1) if resolution_x > 1 else 0
        self.dz = length / (resolution_y - 1) if resolution_y > 1 else 0
        
        self._create_point_masses()
        self._create_springs()
    
    def _create_point_masses(self):
        """Create a grid of point masses"""
        self.point_masses = []
        
        for j in range(self.resolution_y):
            row = []
            for i in range(self.resolution_x):
                # Calculate position
                x = i * self.dx - self.width / 2  # Center the cloth
                y = 0 # Initial height
                z = j * self.dz - self.length / 2  
                
                # Create point mass 
                point_mass = PointMass(x, y, z, self.mass_per_point)
                row.append(point_mass)
            
            self.point_masses.append(row)
    
    def _create_springs(self):
        """Create springs connecting neighboring point masses"""
        self.springs = []
        
        for j in range(self.resolution_y):
            for i in range(self.resolution_x):
                current_point = self.point_masses[j][i]
                
                # Horizontal spring (connect to right neighbor)
                if i < self.resolution_x - 1:
                    right_neighbor = self.point_masses[j][i + 1]
                    spring = Spring(current_point, right_neighbor, self.dx, self.spring_stiffness, self.spring_damping)
                    self.springs.append(spring)
                
                # Vertical spring (connect to bottom neighbor)
                if j < self.resolution_y - 1:
                    bottom_neighbor = self.point_masses[j + 1][i]
                    spring = Spring(current_point, bottom_neighbor, self.dz, self.spring_stiffness, self.spring_damping)
                    self.springs.append(spring)
                
                # Diagonal springs for additional stability (optional but recommended)
                # Diagonal down-right
                if i < self.resolution_x - 1 and j < self.resolution_y - 1:
                    diag_neighbor = self.point_masses[j + 1][i + 1]
                    diag_length = math.sqrt(self.dx**2 + self.dz**2)
                    spring = Spring(current_point, diag_neighbor, diag_length, self.spring_stiffness * 0.5, self.spring_damping)
                    self.springs.append(spring)
                
                # Diagonal down-left
                if i > 0 and j < self.resolution_y - 1:
                    diag_neighbor = self.point_masses[j + 1][i - 1]
                    diag_length = math.sqrt(self.dx**2 + self.dz**2)
                    spring = Spring(current_point, diag_neighbor, diag_length, self.spring_stiffness * 0.5, self.spring_damping)
                    self.springs.append(spring)
    
    def get_all_point_masses(self):
        """Get a flat list of all point masses in the cloth"""
        all_points = []
        for row in self.point_masses:
            all_points.extend(row)
        return all_points
    
    def get_all_springs(self):
        """Get a list of all springs in the cloth"""
        return self.springs
    
    def pin_point(self, i, j):
        """Pin a specific point mass so it doesn't move (useful for hanging cloth)"""
        if 0 <= i < self.resolution_x and 0 <= j < self.resolution_y:
            point = self.point_masses[j][i]
            point.pinned = True
            # Store original position for pinned points
            point.pinned_position = point.position.copy()
    
    def pin_top_edge(self):
        """Pin all points along the top edge of the cloth"""
        for i in range(self.resolution_x):
            self.pin_point(i, 0)
    
    def pin_corners(self):
        """Pin the four corner points of the cloth"""
        self.pin_point(0, 0)  # Top-left
        self.pin_point(self.resolution_x - 1, 0)  # Top-right
        self.pin_point(0, self.resolution_y - 1)  # Bottom-left
        self.pin_point(self.resolution_x - 1, self.resolution_y - 1)  # Bottom-right
    
    def translate(self, x, y, z):
        """Translate all point masses in the cloth by (x, y, z)."""
        offset = np.array([x, y, z], dtype=float)
        for row in self.point_masses:
            for point in row:
                point.position += offset
                # If the point is pinned, update its pinned position as well
                if hasattr(point, 'pinned') and point.pinned:
                    point.pinned_position += offset
    
    def apply_wind_force(self, wind_vector):
        """Apply wind force to all point masses"""
        for row in self.point_masses:
            for point in row:
                if not hasattr(point, 'pinned') or not point.pinned:
                    point.apply_force(wind_vector)
