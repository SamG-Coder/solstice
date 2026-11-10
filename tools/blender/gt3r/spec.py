"""The 911 GT3 R's dimensions: what Porsche publishes, and what was measured from its
photographs (see target/gt3r/REFERENCES.md). Metres. X to the right, Y ahead, Z up, from the
ground under the middle of the wheelbase."""

from common import curve

# Published.
WHEELBASE = 2.507
WIDTH_FRONT, WIDTH_REAR = 2.039, 2.050
# The class's tyres: 300/680-18 at the front, 310/710-18 at the rear (estimated).
FRONT_RADIUS, REAR_RADIUS = 0.340, 0.355
FRONT_TYRE, REAR_TYRE = 0.300, 0.310
RIM_RADIUS = 0.2286
# As in scenes/circuit/gt3r.vehicle, which the engine finds the wheels by.
FRONT_AT = REAR_AT = WHEELBASE / 2
FRONT_TRACK, REAR_TRACK = 1.70, 1.72

HUBS = {
    "wheel_fl": (-FRONT_TRACK / 2, FRONT_AT, FRONT_RADIUS),
    "wheel_fr": (FRONT_TRACK / 2, FRONT_AT, FRONT_RADIUS),
    "wheel_rl": (-REAR_TRACK / 2, -REAR_AT, REAR_RADIUS),
    "wheel_rr": (REAR_TRACK / 2, -REAR_AT, REAR_RADIUS),
}

# Measured from the side view M22_3548.
NOSE, TAIL = 2.172, -2.268
FLOOR = 0.070
SPLITTER_TIP = 2.257
DUCKTAIL_TIP = (-2.336, 0.887)

# ---- The body's lines along the car. Each is a function of Y.

# The top on the middle line: tail, engine cover, rear screen and roof in one arc, windscreen,
# bonnet, nose. From the rear screen to the windscreen's foot it is the measured silhouette.
ZT = curve([(TAIL, 0.56), (-2.25, 0.66), (-2.21, 0.79), (-2.16, 0.862), (-2.10, 0.884),
            (-1.985, 0.870), (-1.89, 0.878), (-1.77, 0.905), (-1.65, 0.948), (-1.53, 0.989),
            (-1.345, 1.050), (-1.16, 1.104), (-0.98, 1.150), (-0.80, 1.195), (-0.61, 1.226),
            (-0.43, 1.241), (-0.25, 1.249), (-0.06, 1.241), (0.12, 1.214), (0.30, 1.157),
            (0.49, 1.073), (0.67, 1.005), (0.85, 0.917), (0.98, 0.850), (1.2, 0.797),
            (1.4, 0.765), (1.6, 0.712), (1.8, 0.634), (1.95, 0.562), (2.05, 0.530),
            (2.13, 0.492), (2.16, 0.44), (NOSE, 0.38)])

# The crown of each wing, and the top of the door between them: its height and how far out.
# Ahead of the windscreen it is the measured silhouette.
ZC = curve([(TAIL, 0.56), (-2.25, 0.66), (-2.15, 0.775), (-2.0, 0.805), (-1.8, 0.83),
            (-1.5, 0.845), (-1.25, 0.85), (-1.0, 0.85), (-0.8, 0.85), (-0.5, 0.85),
            (0.0, 0.845), (0.6, 0.845), (0.94, 0.860), (1.035, 0.856), (1.126, 0.837),
            (1.218, 0.825), (1.309, 0.818), (1.40, 0.810), (1.49, 0.791), (1.584, 0.761),
            (1.675, 0.730), (1.767, 0.673), (1.858, 0.635), (1.95, 0.581), (2.04, 0.547),
            (2.133, 0.495), (2.16, 0.44), (NOSE, 0.38)])
XC = curve([(TAIL, 0.66), (-2.2, 0.76), (-2.0, 0.81), (-1.6, 0.86), (-1.25, 0.87),
            (-0.9, 0.87), (-0.5, 0.865), (0.0, 0.86), (0.5, 0.855), (0.8, 0.84), (1.0, 0.81),
            (1.3, 0.785), (1.6, 0.765), (1.9, 0.735), (2.0, 0.70), (2.1, 0.62), (NOSE, 0.50)])

# The foot of the glass along the cabin; ahead of it the valley between bonnet and wing; behind
# it where the wing's top meets the engine cover.
ZV = curve([(TAIL, 0.55), (-2.25, 0.655), (-2.15, 0.80), (-2.0, 0.862), (-1.6, 0.89),
            (-1.25, 0.885), (-0.8, 0.868), (-0.5, 0.850), (0.0, 0.840), (0.6, 0.840),
            (0.95, 0.842), (1.0, 0.836), (1.2, 0.766), (1.4, 0.727), (1.6, 0.672),
            (1.8, 0.596), (1.95, 0.538), (2.05, 0.522), (2.13, 0.49), (2.16, 0.44),
            (NOSE, 0.38)])
XV = curve([(TAIL, 0.52), (-2.15, 0.64), (-2.0, 0.66), (-1.8, 0.64), (-1.5, 0.63),
            (-1.25, 0.66), (-1.0, 0.72), (-0.8, 0.77), (-0.5, 0.80), (0.0, 0.81),
            (0.5, 0.805), (0.75, 0.79), (0.95, 0.70), (1.1, 0.61), (1.4, 0.565), (1.7, 0.565),
            (1.95, 0.535), (2.1, 0.44), (NOSE, 0.30)])

# The roof's rail over the cabin, and the crest of the windscreen pillar ahead of it; further
# ahead and behind, a line part-way in across bonnet and cover. The pillar's crest and the
# rail's drop are measured: in the side view M22_3548 the windscreen's glass shows 0.075 m
# ahead of the red seal of the door's opening at every height, and the seal itself was found
# pixel by pixel; in the top view M22_3546 the windscreen's lower corners are 0.27 m behind
# its middle.
XR = curve([(TAIL, 0.30), (-2.1, 0.36), (-1.7, 0.42), (-1.4, 0.48), (-1.0, 0.52),
            (-0.6, 0.53), (-0.25, 0.53), (0.1, 0.53), (0.22, 0.54), (0.338, 0.58),
            (0.465, 0.645), (0.60, 0.715), (0.70, 0.74), (0.85, 0.70), (0.98, 0.62),
            (1.1, 0.45), (1.25, 0.33), (1.7, 0.28), (2.1, 0.22), (NOSE, 0.18)])
# How far below the middle line that rail is.
RAIL_DROP = curve([(TAIL, 0.004), (-1.9, 0.005), (-1.65, 0.014), (-1.345, 0.028),
                   (-0.98, 0.030), (-0.6, 0.036), (-0.3, 0.044), (-0.15, 0.052),
                   (0.0, 0.068), (0.073, 0.075), (0.149, 0.080), (0.225, 0.093),
                   (0.30, 0.110), (0.378, 0.120), (0.455, 0.137), (0.527, 0.148),
                   (0.611, 0.166), (0.72, 0.140), (0.85, 0.055), (0.98, 0.012),
                   (1.2, 0.004), (NOSE, 0.002)])

# The widest the body is: over the boxed wings, the published widths; between them, the doors.
# From the top view M22_3546.
XW = curve([(TAIL, 0.74), (-2.24, 0.84), (-2.15, 0.90), (-2.0, 0.935), (-1.8, 0.965),
            (-1.62, 1.0), (-1.45, 1.022), (-1.25, 1.025), (-1.05, 1.022), (-0.93, 0.99),
            (-0.80, 0.948), (-0.5, 0.925), (-0.1, 0.905), (0.2, 0.908), (0.6, 0.905),
            (0.72, 0.905), (0.80, 0.94), (0.885, 1.0), (0.95, 1.014), (1.25, 1.019),
            (1.5, 1.015), (1.70, 1.0), (1.86, 0.970), (1.944, 0.913), (2.03, 0.80),
            (2.11, 0.66), (NOSE, 0.52)])

# The underside: flat, rising behind the rear wheels for the diffuser, and at the nose.
ZB = curve([(TAIL, 0.30), (-2.1, 0.24), (-1.75, 0.12), (-1.45, FLOOR), (1.85, FLOOR),
            (2.05, 0.10), (2.13, 0.16), (NOSE, 0.24)])

# ---- Where things are on the body, from the side view. (Y, Z).
# The side glass: one teardrop from the windscreen's pillar back to a point over the rear wheel.
# Its top and front are the red seal in M22_3548, found by its colour, less 8 mm.
SIDE_GLASS = [(0.565, 0.830), (0.20, 0.842), (-0.30, 0.848), (-0.55, 0.857), (-0.69, 0.873),
              (-0.92, 0.911), (-1.145, 0.965), (-0.841, 1.065), (-0.689, 1.103),
              (-0.536, 1.130), (-0.384, 1.155), (-0.269, 1.162), (-0.155, 1.162),
              (-0.002, 1.140), (0.074, 1.117), (0.150, 1.079), (0.303, 0.993),
              (0.455, 0.895)]
PILLAR_AT = -0.505          # the pillar behind the door, across the glass
DOOR_FRONT, DOOR_REAR = 0.665, -0.525
SIDE_INTAKE = [(-0.565, 0.745), (-0.572, 0.50), (-0.60, 0.33), (-0.70, 0.305), (-0.80, 0.42),
               (-0.845, 0.58), (-0.815, 0.735)]
