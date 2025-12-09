from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Sequence, Tuple

import numpy as np

from highway_env import utils


if TYPE_CHECKING:
    from highway_env.road.lane import AbstractLane
    from highway_env.road.road import Road

LaneIndex = Tuple[str, str, int]


class RoadObject(ABC):
    """
    Common interface for objects that appear on the road.

    For now we assume all objects are rectangular.
    """

    LENGTH: float = 2  # Object length [m]
    WIDTH: float = 2  # Object width [m]

    def __init__(
        self,
        road: Road,
        position: Sequence[float],
        heading: float = 0,
        speed: float = 0,
    ):
        """
        :param road: the road instance where the object is placed in
        :param position: cartesian position of object in the surface
        :param heading: the angle from positive direction of horizontal axis
        :param speed: cartesian speed of object in the surface
        """
        self.road = road
        self.position = np.array(position, dtype=np.float64)
        self.heading = heading
        self.speed = speed
        self.lane_index = (
            self.road.network.get_closest_lane_index(self.position, self.heading)
            if self.road
            else np.nan
        )
        self.lane = self.road.network.get_lane(self.lane_index) if self.road else None

        # Enable collision with other collidables
        self.collidable = True

        # Collisions have physical effects
        self.solid = True

        # If False, this object will not check its own collisions, but it can still collides with other objects that do
        # check their collisions.
        self.check_collisions = True

        self.diagonal = np.sqrt(self.LENGTH**2 + self.WIDTH**2)
        self.crashed = False
        self.hit = False
        self.impact = np.zeros(self.position.shape)
        self.collision_classification: Optional[CollisionClassification] = None

    @classmethod
    def make_on_lane(
        cls,
        road: Road,
        lane_index: LaneIndex,
        longitudinal: float,
        speed: float | None = None,
    ) -> RoadObject:
        """
        Create a vehicle on a given lane at a longitudinal position.

        :param road: a road object containing the road network
        :param lane_index: index of the lane where the object is located
        :param longitudinal: longitudinal position along the lane
        :param speed: initial speed in [m/s]
        :return: a RoadObject at the specified position
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(
            road, lane.position(longitudinal, 0), lane.heading_at(longitudinal), speed
        )

    def handle_collisions(self, other: RoadObject, dt: float = 0) -> None:
        """
        Check for collision with another vehicle.

        :param other: the other vehicle or object
        :param dt: timestep to check for future collisions (at constant velocity)
        """
        if other is self or not (self.check_collisions or other.check_collisions):
            return
        if not (self.collidable and other.collidable):
            return
        intersecting, will_intersect, transition = self._is_colliding(other, dt)
        if will_intersect:
            if self.solid and other.solid:
                if isinstance(other, Obstacle):
                    self.impact = transition
                elif isinstance(self, Obstacle):
                    other.impact = transition
                else:
                    self.impact = transition / 2
                    other.impact = -transition / 2

                # Classify the impending collision
                # Only set if not already classified (to preserve first collision info)
                if self.collision_classification is None:
                    self.collision_classification = classify_collision(
                        self.polygon(), other.polygon(), transition
                    )
                    # Also set for other vehicle (with negated MTV)
                    other.collision_classification = classify_collision(
                        other.polygon(), self.polygon(), -transition
                    )

        if intersecting:
            if self.solid and other.solid:
                self.crashed = True
                other.crashed = True
            if not self.solid:
                self.hit = True
            if not other.solid:
                other.hit = True

            # Classify collision using the already computed MTV (transition)
            # Only set if not already classified (to preserve first collision info)
            if self.collision_classification is None:
                self.collision_classification = classify_collision(
                    self.polygon(), other.polygon(), transition
                )
                # Also set for other vehicle (with negated MTV)
                other.collision_classification = classify_collision(
                    other.polygon(), self.polygon(), -transition
                )

    def _is_colliding(self, other, dt):
        # Fast spherical pre-check
        if (
            np.linalg.norm(other.position - self.position)
            > (self.diagonal + other.diagonal) / 2 + self.speed * dt
        ):
            return (
                False,
                False,
                np.zeros(
                    2,
                ),
            )
        # Accurate rectangular check
        return utils.are_polygons_intersecting(
            self.polygon(), other.polygon(), self.velocity * dt, other.velocity * dt
        )

    # Just added for sake of compatibility
    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        d = {
            "presence": 1,
            "x": self.position[0],
            "y": self.position[1],
            "vx": 0.0,
            "vy": 0.0,
            "cos_h": np.cos(self.heading),
            "sin_h": np.sin(self.heading),
            "cos_d": 0.0,
            "sin_d": 0.0,
        }
        if not observe_intentions:
            d["cos_d"] = d["sin_d"] = 0
        if origin_vehicle:
            origin_dict = origin_vehicle.to_dict()
            for key in ["x", "y", "vx", "vy"]:
                d[key] -= origin_dict[key]
        return d

    @property
    def direction(self) -> np.ndarray:
        return np.array([np.cos(self.heading), np.sin(self.heading)])

    @property
    def velocity(self) -> np.ndarray:
        return self.speed * self.direction

    def polygon(self) -> np.ndarray:
        points = np.array(
            [
                [-self.LENGTH / 2, -self.WIDTH / 2],
                [-self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, +self.WIDTH / 2],
                [+self.LENGTH / 2, -self.WIDTH / 2],
            ]
        ).T
        c, s = np.cos(self.heading), np.sin(self.heading)
        rotation = np.array([[c, -s], [s, c]])
        points = (rotation @ points).T + np.tile(self.position, (4, 1))
        return np.vstack([points, points[0:1]])

    def lane_distance_to(self, other: RoadObject, lane: AbstractLane = None) -> float:
        """
        Compute the signed distance to another object along a lane.

        :param other: the other object
        :param lane: a lane
        :return: the distance to the other other [m]
        """
        if not other:
            return np.nan
        if not lane:
            lane = self.lane
        return (
            lane.local_coordinates(other.position)[0]
            - lane.local_coordinates(self.position)[0]
        )

    @property
    def on_road(self) -> bool:
        """Is the object on its current lane, or off-road?"""
        return self.lane.on_lane(self.position)

    def front_distance_to(self, other: RoadObject) -> float:
        return self.direction.dot(other.position - self.position)

    def __str__(self):
        return f"{self.__class__.__name__} #{id(self) % 1000}: at {self.position}"

    def __repr__(self):
        return self.__str__()


class Obstacle(RoadObject):
    """Obstacles on the road."""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = True


class Landmark(RoadObject):
    """Landmarks of certain areas on the road that must be reached."""

    def __init__(
        self, road, position: Sequence[float], heading: float = 0, speed: float = 0
    ):
        super().__init__(road, position, heading, speed)
        self.solid = False


# Collision Classification Logic

from dataclasses import dataclass
from typing import Optional, List

VERTEX_NAMES = {
    0: "rear-left corner",   
    1: "rear-right corner", 
    2: "front-right corner", 
    3: "front-left corner", 
}

EDGE_NAMES = {
    0: "rear edge",    
    1: "right edge",   
    2: "front edge",   
    3: "left edge",    
}

# Which vertices form which edges
EDGE_VERTICES: Tuple[Tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
)

@dataclass(frozen=True)
class CollisionClassification:
    """Simple collision classification result."""

    contact_type: str  # "edge-edge", "vertex-edge", "vertex-vertex"
    collision_type: str  # "rear-end", "side-swipe", "head-on"
    ego_feature: str  # "front edge", "rear-right corner", etc.
    npc_feature: str
    ego_vertices: Tuple[int, ...]  # Vertex indices
    npc_vertices: Tuple[int, ...]
    ego_edges: Tuple[int, ...]  # Edge indices
    npc_edges: Tuple[int, ...]


def classify_collision(
    ego_polygon: np.ndarray,
    npc_polygon: np.ndarray,
    mtv: np.ndarray,
) -> CollisionClassification:
    """
    Classify collision using the Minimum Translation Vector (MTV) from the SAT check.
    """
    # Get vertices
    ego_verts = _get_vertices(ego_polygon)
    npc_verts = _get_vertices(npc_polygon)

    # The MTV gives us the collision axis and overlap depth
    overlap = np.linalg.norm(mtv)
    if overlap < 1e-10:
        # Should not happen if intersecting is True, but safety check
        axis = np.array([1.0, 0.0])
    else:
        axis = mtv / overlap

    # Project vertices onto this axis to find extremal ones
    ego_proj = ego_verts @ axis
    npc_proj = npc_verts @ axis
    
    # Find extremal vertices on the collision axis
    ego_vertices, npc_vertices = _find_extremal_vertices(ego_proj, npc_proj)

    # Find which edges (if any) are formed by these vertices
    ego_edges = _find_edges(ego_vertices)
    npc_edges = _find_edges(npc_vertices)

    # Classify based on vertex/edge geometry
    contact_type, collision_type, ego_feature, npc_feature = _classify(
        ego_vertices, npc_vertices, ego_edges, npc_edges
    )

    return CollisionClassification(
        contact_type=contact_type,
        collision_type=collision_type,
        ego_feature=ego_feature,
        npc_feature=npc_feature,
        ego_vertices=tuple(ego_vertices),
        npc_vertices=tuple(npc_vertices),
        ego_edges=tuple(ego_edges),
        npc_edges=tuple(npc_edges),
    )


def _get_vertices(polygon: np.ndarray) -> np.ndarray:
    """Extract 4 unique vertices."""
    verts = np.asarray(polygon, dtype=float)
    if verts.shape[0] > 4 and np.allclose(verts[0], verts[-1]):
        verts = verts[:-1]
    if verts.shape[0] != 4:
        raise ValueError(f"Expected 4 vertices, got {verts.shape[0]}")
    return verts


def _find_extremal_vertices(
    ego_proj: np.ndarray,
    npc_proj: np.ndarray,
) -> Tuple[List[int], List[int]]:
    """
    Find extremal vertices on projection axis.
    """
    ego_min, ego_max = float(ego_proj.min()), float(ego_proj.max())
    npc_min, npc_max = float(npc_proj.min()), float(npc_proj.max())

    if ego_min < npc_min:
        # EGO's max side touches NPC's min side
        ego_target = ego_max
        npc_target = npc_min
    else:
        # EGO's min side touches NPC's max side
        ego_target = ego_min
        npc_target = npc_max

    # Find vertices at extremal positions (tight tolerance for zero penetration)
    ego_vertices = _vertices_at_value(ego_proj, ego_target, tolerance=0.001)
    npc_vertices = _vertices_at_value(npc_proj, npc_target, tolerance=0.001)

    return ego_vertices, npc_vertices


def _vertices_at_value(projections: np.ndarray, target: float, tolerance: float) -> List[int]:
    """Return vertex indices whose projection is at the target value."""
    distances = np.abs(projections - target)
    min_dist = float(distances.min())
    return [int(i) for i, d in enumerate(distances) if d <= min_dist + tolerance]


def _find_edges(vertex_ids: List[int]) -> List[int]:
    """Find which edges are formed by the given vertices."""
    if len(vertex_ids) < 2:
        return []
    v_set = set(vertex_ids)
    return [
        edge_id
        for edge_id, (v1, v2) in enumerate(EDGE_VERTICES)
        if v1 in v_set and v2 in v_set
    ]


def _classify(
    ego_v: List[int],
    npc_v: List[int],
    ego_e: List[int],
    npc_e: List[int],
) -> Tuple[str, str, str, str]:
    """Classify collision from vertex/edge indices."""
    # Edge-edge: both vehicles have an edge at contact
    if ego_e and npc_e:
        e_edge, n_edge = ego_e[0], npc_e[0]
        if e_edge == 2 and n_edge == 0:
            coll_type = "rear-end"
        elif e_edge == 0 and n_edge == 2:
            coll_type = "rear-ended"
        elif e_edge == 2 and n_edge == 2:
            coll_type = "head-on"
        else:
            coll_type = "side-swipe"
        return "edge-edge", coll_type, EDGE_NAMES[e_edge], EDGE_NAMES[n_edge]

    # Vertex-edge: NPC vertex hits EGO edge
    if ego_e and npc_v:
        e_edge = ego_e[0]
        n_vertex = npc_v[0]
        if e_edge == 2:
            coll_type = "rear-end"
        elif e_edge == 0:
            coll_type = "rear-ended"
        else:
            coll_type = "side-swipe"
        return "vertex-edge", coll_type, EDGE_NAMES[e_edge], VERTEX_NAMES[n_vertex]

    # Vertex-edge: EGO vertex hits NPC edge
    if npc_e and ego_v:
        n_edge = npc_e[0]
        e_vertex = ego_v[0]
        if n_edge == 0:
            coll_type = "rear-end"
        elif n_edge == 2:
            coll_type = "rear-ended"
        else:
            coll_type = "side-swipe"
        return "vertex-edge", coll_type, VERTEX_NAMES[e_vertex], EDGE_NAMES[n_edge]

    # Vertex-vertex: single vertices touching
    if ego_v and npc_v:
        e_v, n_v = ego_v[0], npc_v[0]
        e_front = e_v in (2, 3)
        n_front = n_v in (2, 3)
        e_left = e_v in (0, 3)
        n_left = n_v in (0, 3)
        if e_left != n_left:
            coll_type = "side-swipe"
        elif e_front and not n_front:
            coll_type = "rear-end"
        elif not e_front and n_front:
            coll_type = "rear-ended"
        elif e_front and n_front:
            coll_type = "head-on"
        else:
            coll_type = "angled"
        return "vertex-vertex", coll_type, VERTEX_NAMES[e_v], VERTEX_NAMES[n_v]

    # Fallback
    return "complex", "angled", f"{len(ego_v)} verts", f"{len(npc_v)} verts"
