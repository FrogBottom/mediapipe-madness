import numpy as np

# At the top of this file are some helper functions that I needed for various things.
# Underneath those are 3 slightly different implementations of the "weighted extended orthogonal procrustes" solver.
# Below those are all the helper functions used by the mediapipe code to to pose estimation, and the convert() function that calls them.

# Take a series of 3D input points, append homogeneous coordinate, perform matrix multipy, and perform perspective divide.
def transform_points(points: np.ndarray, transform: np.ndarray, do_perspective_divide: bool = False) -> np.ndarray:
    points = np.hstack((points,np.ones((points.shape[0],1)))) # Append homogeneous coordinate to each point.
    transformed = points @ transform.T # Multiply. Using the transpose of the matrix lets us keep the output in row-order.
    return (transformed / transformed[:,3][:, np.newaxis])[:,:3] if do_perspective_divide else transformed[:, :3] # Strip off the homogeneous coordinate (or divide by it, if enabled).

# Make a (symmetric) perspective projection matrix from vertical field of view, aspect ratio (x/y) and near+far distance.
def make_frustum(fov_y_rads: float, aspect_ratio: float, near: float, far: float):
    half_tan = np.tan(fov_y_rads / 2)
    top = near * half_tan
    right = top * aspect_ratio
    matrix = np.array([[near / right, 0,          0,                               0                              ],
                       [0,            near / top, 0,                               0                              ],
                       [0,            0,         -(far + near) / (far - near),    -(2 * far * near) / (far - near)],
                       [0,            0,         -1,                               0                              ]])
    return matrix

# Get the near plane distance from camera origin for a given perspective matrix.
def get_near_plane_dist_from_matrix(m: np.ndarray):
    return m[2, 3] / (m[2, 2] - 1)

# Convert a perspective matrix to the opengl-style frustum params, and returns just the ones we happen to care about.
# These are the left, right, bottom, and top distances for the near plane.
def get_near_plane_dims_from_matrix(m: np.ndarray):
    n = get_near_plane_dist_from_matrix(m)
    l = n * (m[0, 2] - 1) / m[0, 0]
    r = n * (m[0, 2] + 1) / m[0, 0]
    b = n * (m[1, 2] - 1) / m[1, 1]
    t = n * (m[1, 2] + 1) / m[1, 1]
    return (l, r), (b, t)

# Convert a 3x3 rotation matrix to euler angles. Only used as a convenience for logging/debugging.
# I don't remember what the order of application for the euler angles is and I can't be bothered to work it out.
# It's either XYZ intrinsic or extrinsic, I dunno.
def rmat_to_euler(rmat):
    b = -np.arcsin(rmat[2, 0])
    cos_b = np.cos(b)
    a = np.arctan2(rmat[2, 1] / cos_b, rmat[2, 2] / cos_b)
    g = np.arctan2(rmat[1, 0] / cos_b, rmat[0, 0] / cos_b)
    return np.array([a, b, g])

# Python rewrite of the mediapipe source, but I made a couple minor tweaks.
def weighted_extended_orthogonal_procrustes3(sources, targets, weights):
    sqrt_weights = np.sqrt(weights)
    weighted_sources_t = (sources.T * sqrt_weights) # transposed(A_w)
    weighted_targets_t = (targets.T * sqrt_weights) # transposed(B_w)
    total_weight = np.sum(sqrt_weights * sqrt_weights) # w
    twice_weighted_sources = weighted_sources_t * sqrt_weights
    source_centroid = np.sum(twice_weighted_sources, axis=1) / total_weight # c_w
    centered_weighted_sources = weighted_sources_t - source_centroid.reshape(-1, 1) * sqrt_weights

    rotation = compute_optimal_rotation(weighted_targets_t @ centered_weighted_sources.T)
    scale = compute_optimal_scale(centered_weighted_sources, weighted_sources_t, weighted_targets_t, rotation)

    pointwise_diffs = weighted_targets_t - (scale * rotation) @ weighted_sources_t
    weighted_pointwise_diffs = pointwise_diffs * sqrt_weights
    translation = np.sum(weighted_pointwise_diffs, axis=1) / total_weight

    result_transform = np.eye(4)
    result_transform[0:3, 0:3] = scale * rotation
    result_transform[0:3, 3] = translation
    result_transform = result_transform

    # Transform all the source points, just for fun.
    source_transformed = transform_points(sources, result_transform)

    # Compute disparity so we can see how good the result was.
    disparity = np.sum((sqrt_weights * sqrt_weights).reshape(-1, 1) * ((source_transformed - targets) ** 2))
    return {
        "scale": scale,
        "rotation": rotation.T,
        "translation": translation,
        "transformed_source_points": source_transformed,
        "disparity": disparity,
        "transform": result_transform
    }

# Python rewrite of the mediapipe source, with minimal changes.
def weighted_extended_orthogonal_procrustes2(sources, targets, weights):
    sqrt_weights = np.sqrt(weights)
    sqrt_weights_t = sqrt_weights.reshape(-1, 1)
    weighted_sources = (sources * sqrt_weights_t) # A_w
    weighted_targets = (targets * sqrt_weights_t) # B_w
    total_weight = np.sum(sqrt_weights * sqrt_weights) # w

    twice_weighted_sources = weighted_sources * sqrt_weights_t
    source_centroid = np.sum(twice_weighted_sources, axis=0) / total_weight # c_w
    centered_weighted_sources = weighted_sources - source_centroid * sqrt_weights_t

    rotation = compute_optimal_rotation(weighted_targets.T @ centered_weighted_sources)
    scale = compute_optimal_scale(centered_weighted_sources.T, weighted_sources.T, weighted_targets.T, rotation)

    pointwise_diffs = weighted_targets.T - (scale * rotation) @ weighted_sources.T
    weighted_pointwise_diffs = pointwise_diffs.T * sqrt_weights_t
    translation = np.sum(weighted_pointwise_diffs, axis=0) / total_weight

    result_transform = np.eye(4)
    result_transform[0:3, 0:3] = scale * rotation
    result_transform[0:3, 3] = translation
    result_transform = result_transform
    # Transform all the source points, just for fun.
    source_transformed = transform_points(sources, result_transform)

    # Compute disparity so we can see how good the result was.
    disparity = np.sum(sqrt_weights_t * sqrt_weights_t * ((source_transformed[:,0:3] - targets) ** 2))
    return {
        "scale": scale,
        "rotation": rotation.T,
        "translation": translation,
        "transformed_source_points": source_transformed,
        "disparity": disparity,
        "transform": result_transform
    }

# @AI(Frog): When I was googling stuff, I was presumptuously offered a "working python solution" to this problem.
# After much cleanup, I must begrudgingly admit that it does technically (almost) work. Not quite though. Might be a handedness issue.
# It's maybe more understandable than the mediapipe source if you aren't following along with the paper, but
# in some respects it takes a slightly less optimal path there.
def weighted_extended_orthogonal_procrustes(source_points, target_points, weights):
    # Coerce the inputs to numpy types, if they weren't already.
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)
    weights = np.asarray(weights)

    # Make sure the weights sum to 1 (unit vector), and put them in the diagonal of a matrix for later.
    weights = weights / np.sum(weights)
    weights_as_column = weights.reshape(-1, 1)
    weights_as_diag = np.diag(weights)

    # Compute the weighted average (centroid) of each list of input points.
    source_centroid = np.average(source_points, axis=0, weights=weights)
    target_centroid = np.average(target_points, axis=0, weights=weights)

    # Adjust source and target so that the centroid is the origin, by subgracting the centroid from each point.
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid

    # Compute the "design matrix" which we will pass to SVD. The derivation of this is in the paper.
    design_matrix = source_centered.T @ (weights_as_column * target_centered)

    # Do SVD, and use the outputs to compute the rotation matrix (this is the orthogonal matrix which gives the problem its name).
    U, _, Vt = np.linalg.svd(design_matrix)
    rotation_matrix = U @ Vt

    # Technically this matrix could encode a reflection, which probably isn't correct, so we'll detect and fixup that case here.
    if np.linalg.det(rotation_matrix) < 0:
        Vt[-1, :] *= -1
        rotation_matrix = U @ Vt # Equation (52) in the paper.

    # Solve for the scale. These should be pretty much equation (54) in the paper.
    var_A = np.sum(weights_as_column * (source_centered ** 2))
    cov_AB = np.trace(source_centered @ rotation_matrix @ target_centered.T @ weights_as_diag)
    scale = cov_AB / var_A

    # 7. Solve for the translation.
    translation = target_centroid - scale * (source_centroid @ rotation_matrix)

    # Transform all the source points, just for fun.
    source_transformed = scale * (source_points @ rotation_matrix) + translation

    # Compute disparity so we can see how good the result was.
    disparity = np.sum(weights_as_column * ((source_transformed - target_points) ** 2))

    # Combine translation, rotation, and scale into a 4x4 matrix.
    result_transform = np.eye(4)
    result_transform[0:3, 0:3] = scale * rotation_matrix
    result_transform[0:3, 3] = translation
    result_transform = result_transform

    return {
        "scale": scale,
        "rotation": rotation_matrix,
        "translation": translation,
        "transformed_source_points": source_transformed,
        "disparity": disparity,
        "transform": result_transform
    }

def compute_optimal_rotation(design_matrix) -> np.ndarray:
    post_rotation, _, pre_rotation = np.linalg.svd(design_matrix) # Note: Unlike Eigen, numpy SVD apparently returns us transposed(V), not V.

    # Disallow reflections by forcing determinant to be +1, if it was -1.
    if np.linalg.det(post_rotation) * np.linalg.det(pre_rotation) < 0:
        post_rotation[-1, :] *= -1 # Flip sign of the last column.
    return post_rotation @ pre_rotation

def compute_optimal_scale(centered_weighted_sources, weighted_sources_t, weighted_targets_t, rotation):
    rotated_centered_weighted_sources = rotation @ centered_weighted_sources
    numerator = np.sum(rotated_centered_weighted_sources * weighted_targets_t)
    denominator = np.sum(centered_weighted_sources * weighted_sources_t)
    return numerator / denominator

def project_xy(projection_matrix: np.ndarray, landmark_points: np.ndarray):
    (left, right), (bottom, top) = get_near_plane_dims_from_matrix(projection_matrix)
    x_scale = right - left
    y_scale = top - bottom

    # Origin of the landmarks is the top left, so we need to flip the Y coordinate here.
    result = np.copy(landmark_points)
    result[:, 1] = 1 - result[:, 1]
    return result * np.array([x_scale, y_scale, x_scale]) + np.array([left, bottom, 0])

def estimate_scale(canonical_landmarks, landmarks, landmark_weights):
    result = weighted_extended_orthogonal_procrustes3(canonical_landmarks, landmarks, landmark_weights)
    return result["scale"]

def move_and_rescale_z(projection_matrix, depth_offset, scale, landmarks):
    result = np.copy(landmarks)
    near = get_near_plane_dist_from_matrix(projection_matrix)
    result[:, 2] = (result[:, 2] - depth_offset + near) / scale
    return result

def unproject_xy(projection_matrix, landmarks):
    result = np.copy(landmarks)
    near= get_near_plane_dist_from_matrix(projection_matrix)
    return np.array([[p[0] * p[2] / near, p[1] * p[2] / near, p[2]] for p in result])

def change_handedness(landmarks):
    result = np.copy(landmarks)
    result[:, 2] *= -1
    return result

# Some random ideas for improvements:
# 1. If it were simple enough to run the procrustes solver with the non-diagonal weights matrix, we could potentially weight the relationship between pairs of landmarks more easily.
# I haven't done the research to see if this is true in reality. If it is, we could possibly use more metric landmarks, by for example including the relationship between
# joints along each finger, but ignoring the relationship between fingers and palm, etc. This would probably at least help with scale estimation.
# 2. Since we estimate scale separately, it might be valuable to run the "first iteration" scale estimation with more (or all) landmarks, and a metric landmark set based on
# the "world" landmarks that the model returns. Or there may be a simpler way to accomplish this. Could improve scale estimation quite a lot.
# 3. In a similar vein, I wonder if we could take advantage of the structure of the hand without having to presume a rigidbody. Translation and scale estimation are pretty good.
# The hard part is rotation. If there were some way to constrain the points such that each (non-thumb) finger's landmarks all exist on one plane, that might help.
# 4. Another thing we could do is handle the depth rewrites in a special way just for hands. Maybe we can use the "world landmarks" just for the Z values?
def convert(runtime_screen_landmarks: np.ndarray, canonical_metric_landmarks: np.ndarray, landmark_weights: np.ndarray, projection_matrix: np.ndarray, do_z_rewrites: bool=True):
    # Step 1. Project landmarks to the frustum near plane. Note that this doesn't mean the Z coordinate is 0. We just move from screen space to one based on the frustum
    # near plane size, and the total scale of Z is assumed to be equal to the total scale of X.
    # This means that we do still have a Z coordinate, and the landmarks are not centered at Z=0 (hence the depth offset).
    screen_landmarks_projected = project_xy(projection_matrix, runtime_screen_landmarks)
    depth_offset = np.mean(screen_landmarks_projected[:, 2]) # Find average Z depth.
    intermediate_landmarks = change_handedness(screen_landmarks_projected) # Landmarks come in left-handed, so we will flip to right-handed before doing any pose estimation.

    # Do the "first iteration" solve. We will throw away everything except the scale, and we will use the estimated scale to try and unproject, so that we have a better
    # guess for the Z coordinates of the landmarks in "screen" space.
    first_result = weighted_extended_orthogonal_procrustes3(canonical_metric_landmarks, intermediate_landmarks, landmark_weights)

    # Unproject back into "screen" space (and switch the handedness again).
    intermediate_landmarks = move_and_rescale_z(projection_matrix, depth_offset, first_result["scale"], intermediate_landmarks)
    intermediate_landmarks = unproject_xy(projection_matrix, intermediate_landmarks)
    intermediate_landmarks = change_handedness(intermediate_landmarks)

    # The mediapipe source only does this for face landmarks, since they don't have a set of "canonical metric landmarks" (basically a source mesh) for hand landmarks.
    # We are experimenting to see if we can get awway with treating the palm landmarks as such a mesh, so if enabled, we do this for hands too.
    if do_z_rewrites:
        # Transform our canonical landmarks (source mesh) using the current estimated transform. Then we can steal the screen-space Z coordinate (depth)
        # of those points. This assumes that transforming the source mesh will probably result in a better depth estimate than we had before.
        z_rewrite_result = weighted_extended_orthogonal_procrustes3(canonical_metric_landmarks, intermediate_landmarks, landmark_weights)
        transformed = transform_points(canonical_metric_landmarks, z_rewrite_result["transform"])
        intermediate_landmarks[:, 2] = transformed[:, 2]

    # Rerun the solver using our new estimated landmark positions, which now includes our estimate of a non-relative Z coordinate.
    second_result = weighted_extended_orthogonal_procrustes3(canonical_metric_landmarks, intermediate_landmarks, landmark_weights)

    # The total scaling factor is the product of the scale estimates from each stage so far.
    total_scale = first_result["scale"] * second_result["scale"]

    # Take the projected screen landmarks, apply our best estimate for Z depth (in the frustum space) and then unproject back to screen space (and switch handedness).
    final_landmarks = move_and_rescale_z(projection_matrix, depth_offset, total_scale, screen_landmarks_projected)
    final_landmarks = unproject_xy(projection_matrix, final_landmarks)
    final_landmarks = change_handedness(final_landmarks)

    # The final landmarks are now in the same coordinate space as the canonical metric landmarks, so we can actually do our final pose estimation to find the transform.
    final_result = weighted_extended_orthogonal_procrustes3(canonical_metric_landmarks, final_landmarks, landmark_weights)

    # OK I lied. If we are rewriting Z coordinates based on the canonical landmarks (source mesh), then we'll do that one more time and re-estimate.
    if do_z_rewrites:
        # Transform the canonical landmarks (source mesh) using our best-guess transform. Then steal the Z coordinates from those.
        # We'll assume those Z coordinates are better than the ones we estimated. Then run pose estimation one last time.
        transformed = transform_points(canonical_metric_landmarks, final_result["transform"])
        final_landmarks[:, 2] = transformed[:, 2]
        final_result = weighted_extended_orthogonal_procrustes3(canonical_metric_landmarks, final_landmarks, landmark_weights)

    # If we want, we can transform the individual landmarks to align them with the "canonical" landmarks (source mesh).
    transform_inv = np.linalg.inv(final_result["transform"])
    runtime_metric_landmarks = transform_points(final_landmarks, transform_inv)
    return final_result, runtime_metric_landmarks
