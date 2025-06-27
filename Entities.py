import math
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from enum import Enum

class IntegrationMethod(Enum):
    EXPLICIT_EULER = 1
    VERLET = 2
    IMPLICIT_EULER = 3
    RUNGE_KUTTA_4 = 4
    POSITION_BASED_DYNAMICS = 5
    
class SpringType(Enum):
    STRUCTURAL = "structural"
    SHEAR = "shear"
    BENDING = "bending"
    
class PointMass:
    def __init__(self, x, y, z, mass=1.0):
        self.position = np.array([x, y, z], dtype=float)
        self.prev_position = self.position.copy()
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.acceleration = np.array([0.0, 0.0, 0.0], dtype=float)
        self.mass = mass
        self.color = (255, 255, 255)
        self.radius = 2 # Visual representation radius
        self.pinned = False  # Whether this point is pinned (fixed in place)
        
    def apply_force(self, force):
        """Apply force to the point mass (F = ma)"""
        self.acceleration += force / self.mass
    
    def apply_verlet_integration(self, dt, vel_damping=0.00025):
        """Update position using Verlet integration"""
        # self.velocity += self.acceleration * dt
        next_position = self.position + (self.position - self.prev_position)*(1-vel_damping) + self.acceleration * dt**2
        temp_position = self.position.copy()  # Store previous position for next update
        self.position = next_position
        self.prev_position = temp_position
        self.velocity = (self.position - self.prev_position) / dt
        self.acceleration.fill(0)  # Reset acceleration
    
    def apply_explicit_euler_integration(self, dt, vel_damping=0.00025):
        """Update position using Euler integration"""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt * (1-vel_damping)
        self.acceleration.fill(0)
    
    def project_3d_to_2d(self, camera_pos, screen_width, screen_height, fov=60):
        """Simple perspective projection"""
        # Translate relative to camera
        rel_pos = self.position - camera_pos
        
        # Avoid division by zero
        if rel_pos[2] <= 0.01:
            rel_pos[2] = 0.01
        
        # Perspective projection
        fov_rad = math.radians(fov)
        f = 1 / math.tan(fov_rad / 2)
        
        x_2d = (rel_pos[0] * f / rel_pos[2]) * screen_width / 2 + screen_width / 2
        y_2d = (rel_pos[1] * f / rel_pos[2]) * screen_height / 2 + screen_height / 2
        
        return int(x_2d), int(y_2d), rel_pos[2]
    
class Spring:
    def __init__(self, point_a: PointMass, point_b: PointMass, rest_length: float, spring_type: SpringType, stiffness=1.0, damping=-.1):
        self.point_a = point_a
        self.point_b = point_b
        self.rest_length = rest_length
        self.stiffness = stiffness
        self.damping = damping
        self.type = spring_type
        # Small epsilon to avoid division by zero
        self.epsilon = 1e-10
        
    def apply_spring_force(self):
        """Optimized spring force calculation with reduced operations"""
        # Calculate displacement vector
        displacement = self.point_b.position - self.point_a.position
        
        # Calculate squared length and early exit
        length_squared = np.dot(displacement, displacement)
        if length_squared < self.epsilon:
            return
        
        # Calculate current length and inverse
        current_length = np.sqrt(length_squared)
        inv_length = 1.0 / current_length
        
        # Calculate the normalized direction vector
        direction = displacement / current_length
        # Spring force magnitude
        force_magnitude = self.stiffness * (current_length - self.rest_length)
        # Damping force magnitude
        force_vector = force_magnitude * direction
        relative_velocity = self.point_b.velocity - self.point_a.velocity
        damping_force = self.damping * np.dot(relative_velocity, direction) * direction

        # Total force
        total_force = force_vector + (-damping_force)
        
        # Apply forces directly to accelerations
        self.point_a.acceleration += total_force / self.point_a.mass
        self.point_b.acceleration -= total_force / self.point_b.mass


    # def apply_spring_force(self):
    #     """Calculate and apply spring force to both connected point masses"""
    #     # Calculate the vector from point_a to point_b
    #     displacement = self.point_b.position - self.point_a.position
        
    #     # Calculate current length of the spring
    #     current_length = np.linalg.norm(displacement)
        
    #     # Avoid division by zero
    #     if current_length == 0:
    #         return
        
    #     # Calculate the normalized direction vector
    #     direction = displacement / current_length
        
    #     # Calculate the spring force magnitude using Hooke's law: F = -k * (x - x0)
    #     # where x is current length, x0 is rest length, k is stiffness
    #     force_magnitude = self.stiffness * (current_length - self.rest_length)
        
    #     # Calculate force vector (pointing from longer to shorter length)
    #     force_vector = force_magnitude * direction
    #     relative_velocity = self.point_b.velocity - self.point_a.velocity
    #     damping_force = self.damping * np.dot(relative_velocity, direction) * direction

    #     # Total force
    #     total_force = force_vector + (-damping_force)
        
    #     # Apply equal and opposite forces to both point masses
    #     self.point_a.apply_force(total_force)   # Force pulls point_a toward point_b
    #     self.point_b.apply_force(-total_force)  # Force pulls point_b toward point_a

class Cloth:
    def __init__(self, width, length, resolution_x, resolution_y, mass_per_point=1.0,
                 structural_spring_stiffness=100.0, shear_spring_stiffness=50.0,
                 bending_spring_stiffness=25.0, spring_damping=.1, integration_method=IntegrationMethod.VERLET,
                 velocity_damping=0.00025):
        """
        Create a cloth as a grid of point masses connected by springs.
        Structural springs connect adjacent points.
        Shear springs connect diagonal neighbors.
        Bending springs connect points two apart.
        
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
        self.structural_spring_stiffness = structural_spring_stiffness
        self.shear_spring_stiffness = shear_spring_stiffness
        self.bending_spring_stiffness = bending_spring_stiffness
        self.spring_damping = spring_damping
        self.integration_method = integration_method
        self.point_masses = []
        self.structural_springs = []
        self.shear_springs = []
        self.bend_springs = []
        self.springs = []  # All springs combined for easy access
        # Flat list of points and mapping for matrix assembly
        self.flat_points = []
        self.point_map = {}
        # Calculate spacing between points
        self.dx = width / (resolution_x - 1) if resolution_x > 1 else 0
        self.dz = length / (resolution_y - 1) if resolution_y > 1 else 0
        self.velocity_damping = velocity_damping #multiply by this to dampen velocity each timestep
        self._create_point_masses()
        self._create_springs()
    
    def _create_point_masses(self):
        """Create a grid of point masses and the flat list/map."""
        self.point_masses = []
        self.flat_points = []
        self.point_map = {}
        idx = 0
        
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

                self.flat_points.append(point_mass)
                self.point_map[point_mass] = idx
                idx += 1
            
            self.point_masses.append(row)
    
    def _create_springs(self):
        
        for j in range(self.resolution_y):
            for i in range(self.resolution_x):
                current_point = self.point_masses[j][i]
                
                # structural springs (connect to right neighbor)
                if i < self.resolution_x - 1:
                    right_neighbor = self.point_masses[j][i + 1]
                    spring = Spring(current_point, right_neighbor, self.dx, SpringType.STRUCTURAL, self.structural_spring_stiffness, self.spring_damping)
                    self.structural_springs.append(spring)
                
                # structural springs (connect to bottom neighbor)
                if j < self.resolution_y - 1:
                    bottom_neighbor = self.point_masses[j + 1][i]
                    spring = Spring(current_point, bottom_neighbor, self.dz, SpringType.STRUCTURAL, self.structural_spring_stiffness, self.spring_damping)
                    self.structural_springs.append(spring)
                
                # shear springs (diagonal down-right)
                if i < self.resolution_x - 1 and j < self.resolution_y - 1:
                    diag_neighbor = self.point_masses[j + 1][i + 1]
                    diag_length = math.sqrt(self.dx**2 + self.dz**2)
                    spring = Spring(current_point, diag_neighbor, diag_length, SpringType.SHEAR, self.shear_spring_stiffness, self.spring_damping)
                    self.shear_springs.append(spring)
                
                # shear springs (diagonal down-left)
                if i > 0 and j < self.resolution_y - 1:
                    diag_neighbor = self.point_masses[j + 1][i - 1]
                    diag_length = math.sqrt(self.dx**2 + self.dz**2)
                    spring = Spring(current_point, diag_neighbor, diag_length, SpringType.SHEAR, self.shear_spring_stiffness, self.spring_damping)
                    self.shear_springs.append(spring)
                    
                # bend springs (2 apart in x direction)
                if i < self.resolution_x - 2:
                    right_neighbor = self.point_masses[j][i + 2]
                    spring = Spring(current_point, right_neighbor, 2*self.dx, SpringType.BENDING, self.bending_spring_stiffness, self.spring_damping)
                    self.bend_springs.append(spring)
                
                # bend springs (2 apart in y direction)
                if j < self.resolution_y - 2:
                    bottom_neighbor = self.point_masses[j + 2][i]
                    spring = Spring(current_point, bottom_neighbor, 2*self.dz, SpringType.BENDING, self.bending_spring_stiffness, self.spring_damping)
                    self.bend_springs.append(spring)
        self.springs = self.structural_springs + self.shear_springs + self.bend_springs
    
    def get_all_point_masses(self):
        """Get a flat list of all point masses in the cloth"""
        return self.flat_points
    
    def timestep(self, dt):
        for spring in self.springs:
            spring.apply_spring_force()
        
        # Update point masses
        for point_mass in self.flat_points:
            #Skip pinned points - they don't move
            if point_mass.pinned:
                continue            
            #Apply gravity
            gravity = np.array([0.0, -9.81, 0.0], dtype=float)  # Gravity vector
            point_mass.apply_force(gravity * point_mass.mass)
            if self.integration_method == IntegrationMethod.VERLET:
                point_mass.apply_verlet_integration(dt, vel_damping=self.velocity_damping)
            elif self.integration_method == IntegrationMethod.EXPLICIT_EULER:
                point_mass.apply_explicit_euler_integration(dt, vel_damping=self.velocity_damping)
            #Velocity clamping
            # velocity_magnitude = np.linalg.norm(point_mass.velocity)
            # if velocity_magnitude > max_velocity:
            #     point_mass.velocity = point_mass.velocity * (max_velocity / velocity_magnitude)
    
            
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
    
    def pin_top_corners(self):
        """Pin the top corner points of the cloth"""
        corner_pts = 6
        for i in range(corner_pts):
            self.pin_point(i, 0)  # Top-left
            self.pin_point(self.resolution_x-1-i, 0)  # Top-right
    
    # def translate(self, x, y, z):
    #     """Translate all point masses in the cloth by (x, y, z)."""
    #     offset = np.array([x, y, z], dtype=float)
    #     for row in self.point_masses:
    #         for point in row:
    #             point.position += offset
    #             # If the point is pinned, update its pinned position as well
    #             if hasattr(point, 'pinned') and point.pinned:
    #                 point.pinned_position += offset
    
    def translate(self, offsetArray: np.ndarray):
        """Translate all point masses in the cloth by a 3 dimensional position offset array."""
        for row in self.point_masses:
            for point in row:
                point.position += offsetArray
                point.prev_position += offsetArray  # Update previous position for Verlet integration
                
    
    def rotate(self, x, y, z):
        """
        Rotate the cloth around its center by x, y, z radians (Euler angles).
        Args:
            x: Rotation angle around X axis (pitch)
            y: Rotation angle around Y axis (yaw)
            z: Rotation angle around Z axis (roll)
        """
        # Compute center of the cloth
        center = np.array([0.0, 0.0, 0.0])
        count = 0
        for row in self.point_masses:
            for point in row:
                center += point.position
                count += 1
        center /= count
    
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(x), -np.sin(x)],
            [0, np.sin(x), np.cos(x)]
        ])
        Ry = np.array([
            [np.cos(y), 0, np.sin(y)],
            [0, 1, 0],
            [-np.sin(y), 0, np.cos(y)]
        ])
        Rz = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])
        # Combined rotation: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
    
        for row in self.point_masses:
            for point in row:
                rel_pos = point.position - center
                point.position = center + R @ rel_pos
    
    def apply_wind_force(self, wind_vector):
        pass 
                    
class SolidObject:
    def __init__(self, position, velocity=None):
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity if velocity else [0.0, 0.0, 0.0], dtype=float)
        self.color = (255, 255, 255)
    
    def check_collision(self, point_mass):
        """Check if a point mass is colliding with this object"""
        raise NotImplementedError("Subclasses must implement collision detection")
    
    def resolve_collision(self, point_mass):
        """Resolve collision with a point mass"""
        raise NotImplementedError("Subclasses must implement collision resolution")
    
    def update(self, dt):
        """Update object position (for moving objects)"""
        self.position += self.velocity * dt

class Sphere(SolidObject):
    def __init__(self, position, radius, velocity=None, restitution=0.1, friction=0.1):
        super().__init__(position, velocity)
        self.radius = radius
        self.restitution = restitution  # Bounciness (0 = no bounce, 1 = perfect bounce)
        self.friction = friction
        self.color = (0, 0, 0)  # Red color for visibility
    
    def check_collision(self, point_mass):
        """Check if point mass is inside or touching the sphere"""
        distance_vector = point_mass.position - self.position
        distance = np.linalg.norm(distance_vector)
        return distance <= self.radius
    
    def resolve_collision(self, point_mass):
        """Push point mass out of sphere and apply collision response"""
        distance_vector = point_mass.position - self.position
        distance = np.linalg.norm(distance_vector)
        
        if distance <= self.radius and distance > 0:
            # Normalize the distance vector to get collision normal
            normal = distance_vector / distance
            
            # Push point mass to surface of sphere
            penetration_depth = self.radius - distance
            point_mass.position = self.position + normal * self.radius
            
            # Calculate relative velocity
            relative_velocity = point_mass.velocity - self.velocity
            velocity_along_normal = np.dot(relative_velocity, normal)
            
            # Don't resolve if velocities are separating
            if velocity_along_normal > 0:
                return
            
            # Calculate restitution
            impulse_scalar = -(1 + self.restitution) * velocity_along_normal
            impulse = impulse_scalar * normal
            
            # Apply impulse (assuming infinite mass for sphere)
            point_mass.velocity += impulse
            
            # Apply friction
            tangent_velocity = relative_velocity - velocity_along_normal * normal
            if np.linalg.norm(tangent_velocity) > 0:
                friction_impulse = -self.friction * np.linalg.norm(impulse) * (tangent_velocity / np.linalg.norm(tangent_velocity))
                point_mass.velocity += friction_impulse

class Plane(SolidObject):
    def __init__(self, position, normal, restitution=0.3, friction=0.2):
        super().__init__(position)
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
        self.restitution = restitution
        self.friction = friction
        self.color = (100, 255, 100)  # Green color
    
    def check_collision(self, point_mass):
        """Check if point mass is below/behind the plane"""
        to_point = point_mass.position - self.position
        distance_to_plane = np.dot(to_point, self.normal)
        return distance_to_plane <= 0
    
    def resolve_collision(self, point_mass):
        """Push point mass above plane and apply collision response"""
        to_point = point_mass.position - self.position
        distance_to_plane = np.dot(to_point, self.normal)
        
        if distance_to_plane <= 0:
            # Push point mass to plane surface
            point_mass.position = point_mass.position - distance_to_plane * self.normal
            
            # # Calculate relative velocity
            # velocity_along_normal = np.dot(point_mass.velocity, self.normal)
            
            # # Don't resolve if velocity is away from plane
            # if velocity_along_normal > 0:
            #     return
            
            # # Apply restitution
            # normal_velocity = velocity_along_normal * self.normal
            # tangent_velocity = point_mass.velocity - normal_velocity
            
            # # New velocity after collision
            # point_mass.velocity = tangent_velocity - self.restitution * normal_velocity
            
            # # Apply friction
            # if np.linalg.norm(tangent_velocity) > 0:
            #     friction_force = -self.friction * tangent_velocity
            #     point_mass.velocity += friction_force

class Box(SolidObject):
    def __init__(self, position, dimensions, velocity=None, restitution=0.0, friction=0.15):
        super().__init__(position, velocity)
        self.dimensions = np.array(dimensions, dtype=float)  # [width, height, depth]
        self.restitution = restitution
        self.friction = friction
        self.color = (100, 100, 255)  # Blue color
    
    def check_collision(self, point_mass):
        """Check if point mass is inside the box"""
        rel_pos = point_mass.position - self.position
        half_dims = self.dimensions / 2
        
        return (abs(rel_pos[0]) <= half_dims[0] and 
                abs(rel_pos[1]) <= half_dims[1] and 
                abs(rel_pos[2]) <= half_dims[2])
    
    def resolve_collision(self, point_mass):
        """Push point mass out of box along shortest path"""
        if not self.check_collision(point_mass):
            return
            
        rel_pos = point_mass.position - self.position
        half_dims = self.dimensions / 2
        
        # Calculate penetration depths for each axis
        penetration_x = half_dims[0] - abs(rel_pos[0])
        penetration_y = half_dims[1] - abs(rel_pos[1])
        penetration_z = half_dims[2] - abs(rel_pos[2])
        
        # Find the axis with minimum penetration (shortest exit path)
        min_penetration = min(penetration_x, penetration_y, penetration_z)
        
        # Determine exit direction and normal
        if min_penetration == penetration_x:
            # Exit through X face
            normal = np.array([1.0 if rel_pos[0] >= 0 else -1.0, 0.0, 0.0])
            point_mass.position[0] = self.position[0] + normal[0] * half_dims[0]
        elif min_penetration == penetration_y:
            # Exit through Y face
            normal = np.array([0.0, 1.0 if rel_pos[1] >= 0 else -1.0, 0.0])
            point_mass.position[1] = self.position[1] + normal[1] * half_dims[1]
        else:
            # Exit through Z face
            normal = np.array([0.0, 0.0, 1.0 if rel_pos[2] >= 0 else -1.0])
            point_mass.position[2] = self.position[2] + normal[2] * half_dims[2]
        
        # Apply collision response
        self._apply_collision_response(point_mass, normal)
    
    def _apply_collision_response(self, point_mass, normal):
        """Apply collision response with restitution and friction"""
        # Calculate relative velocity
        relative_velocity = point_mass.velocity - self.velocity
        velocity_along_normal = np.dot(relative_velocity, normal)
        
        # Only resolve if moving into the surface
        if velocity_along_normal >= 0:
            return
        
        # Calculate collision impulse
        impulse_magnitude = -(1 + self.restitution) * velocity_along_normal
        impulse = impulse_magnitude * normal
        
        # Apply impulse to velocity
        point_mass.velocity += impulse
        
        # Apply friction
        if self.friction > 0:
            # Calculate tangential velocity
            tangent_velocity = relative_velocity - velocity_along_normal * normal
            tangent_speed = np.linalg.norm(tangent_velocity)
            
            if tangent_speed > 0:
                # Calculate friction impulse
                friction_magnitude = min(self.friction * abs(impulse_magnitude), tangent_speed)
                friction_impulse = -friction_magnitude * (tangent_velocity / tangent_speed)
                point_mass.velocity += friction_impulse

    def get_bounds(self):
        """Get the min and max bounds of the box for debugging"""
        half_dims = self.dimensions / 2
        min_bounds = self.position - half_dims
        max_bounds = self.position + half_dims
        return min_bounds, max_bounds
    
    def is_point_inside(self, point):
        """Check if a 3D point is inside the box (useful for debugging)"""
        rel_pos = point - self.position
        half_dims = self.dimensions / 2
        return (abs(rel_pos[0]) <= half_dims[0] and 
                abs(rel_pos[1]) <= half_dims[1] and 
                abs(rel_pos[2]) <= half_dims[2])
    
    def get_closest_surface_point(self, point):
        """Get the closest point on the box surface to a given point"""
        rel_pos = point - self.position
        half_dims = self.dimensions / 2
        
        # Clamp the relative position to the box bounds
        clamped_pos = np.clip(rel_pos, -half_dims, half_dims)
        
        # If the point is inside, project to the nearest face
        if self.is_point_inside(point):
            # Find which face is closest
            distances_to_faces = half_dims - np.abs(clamped_pos)
            min_axis = np.argmin(distances_to_faces)
            
            # Project to that face
            clamped_pos[min_axis] = half_dims[min_axis] * np.sign(rel_pos[min_axis])
        
        return self.position + clamped_pos

# class Box(SolidObject):
#     def __init__(self, position, dimensions, velocity=None, restitution=0.0, friction=0.15):
#         super().__init__(position, velocity)
#         self.dimensions = np.array(dimensions, dtype=float)  # [width, height, depth]
#         self.restitution = restitution
#         self.friction = friction
#         self.color = (100, 100, 255)  # Blue color
    
#     def check_collision(self, point_mass):
#         """Check if point mass is inside the box"""
#         rel_pos = point_mass.position - self.position
#         half_dims = self.dimensions / 2
        
#         return (abs(rel_pos[0]) <= half_dims[0] and 
#                 abs(rel_pos[1]) <= half_dims[1] and 
#                 abs(rel_pos[2]) <= half_dims[2])
    
#     def resolve_collision(self, point_mass):
#         """Push point mass out of box along shortest path"""
#         rel_pos = point_mass.position - self.position
#         half_dims = self.dimensions / 2
        
#         if self.check_collision(point_mass):
#             # Find the axis with minimum penetration
#             penetrations = half_dims - np.abs(rel_pos)
#             min_axis = np.argmin(penetrations)
            
#             # Push out along the axis with minimum penetration
#             push_direction = np.zeros(3)
#             push_direction[min_axis] = 1.0 if rel_pos[min_axis] >= 0 else -1.0
            
#             # Move point mass to surface
#             point_mass.position = self.position + np.sign(rel_pos) * half_dims
#             point_mass.position[min_axis] = self.position[min_axis] + push_direction[min_axis] * half_dims[min_axis]
            
#             # Apply collision response along the normal
#             normal = push_direction
#             relative_velocity = point_mass.velocity - self.velocity
#             velocity_along_normal = np.dot(relative_velocity, normal)
            
#             if velocity_along_normal < 0:  # Moving into the surface
#                 # Apply restitution
#                 impulse = -(1 + self.restitution) * velocity_along_normal * normal
#                 point_mass.velocity += impulse
                
#                 # Apply friction
#                 tangent_velocity = relative_velocity - velocity_along_normal * normal
#                 if np.linalg.norm(tangent_velocity) > 0:
#                     friction_impulse = -self.friction * np.linalg.norm(impulse) * (tangent_velocity / np.linalg.norm(tangent_velocity))
#                     point_mass.velocity += friction_impulse
