'''
    Initially written by Ming Hsiao in MATLAB
    Adapted to Python by Akash Sharma (akashsharma@cmu.edu), 2020
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True, threshold=np.inf, linewidth=np.inf)


def draw_cov_ellipse(mu, cov, color):
    """
    Draws an ellipse in plt canvas.

    \param mu Mean of a Gaussian
    \param cov Covariance of a Gaussian
    \param color Color in plt format, e.g. 'b' for blue, 'r' for red.
    """
    U, s, Vh = np.linalg.svd(cov)
    a, b = s[0], s[1]
    vx, vy = U[0, 0], U[0, 1]
    theta = np.arctan2(vy, vx)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    phi = np.arange(0, 2 * np.pi, np.pi / 50)
    rot = []
    for i in range(100):
        rect = (np.array(
            [3 * np.sqrt(a) * np.cos(phi[i]),
             3 * np.sqrt(b) * np.sin(phi[i])]))[:, None]
        rot.append(R @ rect + mu)

    rot = np.asarray(rot)
    plt.plot(rot[:, 0], rot[:, 1], c=color, linewidth=0.75)


def draw_traj_and_pred(X, P):
    """ Draw trajectory for Predicted state and Covariance

    :X: Prediction vector
    :P: Prediction Covariance matrix
    :returns: None

    """
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'm')
    plt.draw()
    # plt.waitforbuttonpress(0)
    plt.pause(0.1)


def draw_traj_and_map(X, last_X, P, t):
    """Draw Trajectory and map

    :X: Current state
    :last_X: Previous state
    :P: Covariance
    :t: timestep
    :returns: None

    """
    plt.ion()
    draw_cov_ellipse(X[0:2], P[0:2, 0:2], 'b')
    plt.plot([last_X[0], X[0]], [last_X[1], X[1]], c='b', linewidth=0.75)
    plt.plot(X[0], X[1], '*b')

    if t == 0:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + k * 2:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'r')
    else:
        for k in range(6):
            draw_cov_ellipse(
                X[3 + k * 2:3 + k * 2 + 2], P[3 + 2 * k:3 + 2 * k + 2,
                                              3 + 2 * k:3 + 2 * k + 2], 'g')

    plt.draw()
    # plt.waitforbuttonpress(0)
    plt.pause(0.1)


def warp2pi(angle_rad):
    """
    TODO: warps an angle in [-pi, pi]. Used in the update step.

    \param angle_rad Input angle in radius
    \return angle_rad_warped Warped angle to [-\pi, \pi].
    """
    
    angle_rad = np.mod(angle_rad + 3 * np.pi, 2 * np.pi) - np.pi
    
    return angle_rad


def init_landmarks(init_measure, init_measure_cov, init_pose, init_pose_cov):
    '''
    TODO: initialize landmarks given the initial poses and measurements with their covariances
    \param init_measure Initial measurements in the form of (beta0, l0, beta1, l1, ...).
    \param init_measure_cov Initial covariance matrix of shape (2, 2) per landmark given parameters.
    \param init_pose Initial pose vector of shape (3, 1).
    \param init_pose_cov Initial pose covariance of shape (3, 3) given parameters.

    \return k Number of landmarks.
    \return landmarks Numpy array of shape (2k, 1) for the state.
    \return landmarks_cov Numpy array of shape (2k, 2k) for the uncertainty.
    '''

    k = init_measure.shape[0] // 2

    landmark = np.zeros((2 * k, 1))
    landmark_cov = np.zeros((2 * k, 2 * k))
    x = init_pose[0, 0]
    y = init_pose[1, 0]
    theta = init_pose[2, 0]
    
    for i in range(k):
        beta = init_measure[2 * i, 0]
        r = init_measure[2 * i + 1, 0]
        
        beta = warp2pi(beta + init_pose[2, 0])
        landmark[2 * i, 0]     = x + r * np.cos(beta)
        landmark[2 * i + 1, 0] = y + r * np.sin(beta)
        J = np.array(
            [
                [-r * np.sin(beta + theta), np.cos(beta + theta)],
                [r * np.cos(beta + theta), np.sin(beta + theta)]
            ]
        )
        
        landmark_cov[2 * i:2 * i + 2, 2 * i:2 * i + 2] = J @ init_measure_cov @ J.T
    

    return k, landmark, landmark_cov


def predict(X, P, control, control_cov, k):
    '''
    TODO: predict step in EKF SLAM with derived Jacobians.
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.
    \param control Control signal of shape (2, 1) in the polar space that moves the robot.
    \param control_cov Control covariance of shape (3, 3) in the (x, y, theta) space given the parameters.
    \param k Number of landmarks.

    \return X_pre Predicted X state of shape (3 + 2k, 1).
    \return P_pre Predicted P covariance of shape (3 + 2k, 3 + 2k).
    '''

    d = control[0, 0]
    alpha = control[1, 0]
    theta = X[2, 0]
    
    X[0, 0] += d * np.cos(theta)
    X[1, 0] += d * np.sin(theta)
    X[2, 0] += (alpha)
    X[2, 0] = warp2pi(X[2, 0])
    
    
    G = np.eye((3 + 2 * k) )
    G[:3, :3] = np.array([
        [1, 0, -d * np.sin(theta)],
        [0, 1, d * np.cos(theta)],
        [0, 0, 1]
    ])
    
    
    R = np.zeros((3 + 2 * k, 3 + 2 * k))
    Hr = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ]) 
    
    # by moving through the environment, we do not update the covariance of the landmarks
    R[:3, :3] = Hr @ control_cov @ Hr.T 
    P = G @ P @ G.T + R
    return X, P


def update(X_pre, P_pre, measure, measure_cov, k):
    '''
    TODO: update step in EKF SLAM with derived Jacobians.
    \param X_pre Predicted state vector of shape (3 + 2k, 1) from the predict step.
    \param P_pre Predicted covariance matrix of shape (3 + 2k, 3 + 2k) from the predict step.
    \param measure Measurement signal of shape (2k, 1).
    \param measure_cov Measurement covariance of shape (2, 2) per landmark given the parameters.
    \param k Number of landmarks.

    \return X Updated X state of shape (3 + 2k, 1).
    \return P Updated P covariance of shape (3 + 2k, 3 + 2k).
    '''
    Ht = np.zeros((2 * k, 3 + 2 * k))
    Qt = np.zeros((2 * k, 2 * k))
    Hu = np.zeros((2 * k, 1))
    X_u = X_pre.copy()

    for i in range(k):
        delx = X_pre[3 + 2 * i, 0] - X_pre[0, 0]
        dely = X_pre[3 + 2 * i + 1, 0] - X_pre[1, 0]
        
        q = delx**2 + dely**2
        sq = np.sqrt(q)

        Ht[i * 2: i * 2 + 2, :3] = np.array([
            [dely / q, -delx / q, -1],
            [-delx / sq, -dely / sq, 0]
        ])
        
        Ht[i * 2: i * 2 + 2, (3 + 2 * i): (3 + 2 * i) + 2] = np.array([
            [-dely / q, delx / q],
            [delx / sq, dely / sq],
        ])
        Qt[i * 2: i * 2 + 2, i * 2: i * 2 + 2] = measure_cov
        Hu[i * 2: i * 2 + 2, 0] = np.array([
            warp2pi(np.arctan2(dely, delx) - X_pre[2, 0]),
            sq,
        ])        

    K = P_pre @ Ht.T @ np.linalg.inv((Ht @ P_pre @ Ht.T) + Qt)
    
    X_u = X_pre + K @ (measure - Hu)
    P_u = (np.eye(3 + 2 * k) - K @ Ht) @ P_pre
    

    return X_u, P_u


def evaluate(X, P, k):
    '''
    TODO: evaluate the performance of EKF SLAM.
    1) Plot the results.
    2) Compute and print the Euclidean and Mahalanobis distance given X, P, and the ground truth (provided in the function).
    \param X State vector of shape (3 + 2k, 1) stacking pose and landmarks.
    \param P Covariance matrix of shape (3 + 2k, 3 + 2k) for X.

    \return None
    '''
    l_true = np.array([3, 6, 3, 12, 7, 8, 7, 14, 11, 6, 11, 12], dtype=float)
    dists = []
    for i in range(k):
        delx = l_true[2 * i] - X[3 + 2 * i]
        dely = l_true[2 * i + 1] - X[3 + 2 * i + 1]
        
        edist = np.sqrt(delx**2 + dely**2).squeeze()
        a = np.array([delx, dely]).T
        mdist = np.sqrt(a @ np.linalg.inv(P[3 + 2 * i:3 + 2 * i + 2, 3 + 2 * i:3 + 2 * i + 2]) @ a.T)
        mdist = mdist.squeeze()
        
        dists.append((edist, mdist))
        print(f'Euclidean distance for landmark {i}: {edist}')
        print(f"Mahalanobis distance for landmark {i}: {mdist}")
        
    plt.scatter(l_true[0::2], l_true[1::2])
    plt.draw()
    return dists
    # plt.waitforbuttonpress(0)

        


def main():
    # TEST: Setup uncertainty parameters

    for scale in [1, 0.1, 10]:
        for i in range(5):
            if i == 1 and scale == 1:
                break
            kernel = [1, 1, 1, 1, 1]
            kernel[i] = scale
            sig_x = 0.25 * kernel[0]
            sig_y = 0.1 * kernel[1]
            sig_alpha = 0.1  * kernel[2]
            sig_beta = 0.01* kernel[3]
            sig_r = 0.08 * kernel[4]
            
            # Generate variance from standard deviation
            sig_x2 = sig_x**2
            sig_y2 = sig_y**2
            sig_alpha2 = sig_alpha**2
            sig_beta2 = sig_beta**2
            sig_r2 = sig_r**2

            # Open data file and read the initial measurements
            data_file = open("/home/praveenvnktsh/slam_a2/data/data.txt")
            line = data_file.readline()
            fields = re.split('[\t ]', line)[:-1]
            arr = np.array([float(field) for field in fields])
            measure = np.expand_dims(arr, axis=1)
            t = 1

            # Setup control and measurement covariance
            control_cov = np.diag([sig_x2, sig_y2, sig_alpha2])
            measure_cov = np.diag([sig_beta2, sig_r2])

            # Setup the initial pose vector and pose uncertainty
            pose = np.zeros((3, 1))
            pose_cov = np.diag([0.02**2, 0.02**2, 0.1**2])

            ##########
            # TODO: initialize landmarks
            k, landmark, landmark_cov = init_landmarks(measure, measure_cov, pose,
                                                    pose_cov)

            # Setup state vector X by stacking pose and landmark states
            # Setup covariance matrix P by expanding pose and landmark covariances
            X = np.vstack((pose, landmark))
            P = np.block([[pose_cov, np.zeros((3, 2 * k))],
                        [np.zeros((2 * k, 3)), landmark_cov]])

            # Plot initial state and covariance
            last_X = X.copy()
            draw_traj_and_map(X, last_X, P, 0)

            # Core loop: sequentially process controls and measurements
            for line in data_file:
                fields = re.split('[\t ]', line)[:-1]
                arr = np.array([float(field) for field in fields])

                # Control
                if arr.shape[0] == 2:
                    print(f'{t}: Predict step')
                    d, alpha = arr[0], arr[1]
                    control = np.array([[d], [alpha]])

                    ##########
                    # TODO: predict step in EKF SLAM
                    X_pre, P_pre = predict(X, P, control, control_cov, k)

                    draw_traj_and_pred(X_pre, P_pre)

                # Measurement
                else:
                    print(f'{t}: Update step')
                    measure = np.expand_dims(arr, axis=1)

                    ##########
                    # TODO: update step in EKF SLAM
                    X, P = update(X_pre, P_pre, measure, measure_cov, k)
                    draw_traj_and_map(X, last_X, P, t)
                    last_X = X.copy()
                    t += 1

            # EVAL: Plot ground truth landmarks and analyze distances
            dists = evaluate(X, P, k)
            
            import os
            params = f'{round(sig_x, 3)}_{round(sig_y, 3)}_{round(sig_alpha, 3)}_{round(sig_beta, 3)}_{round(sig_r, 3)}'
            os.makedirs('results/' + params, exist_ok=True)
            plt.savefig(f'results/{params}/plot.png')
            with open(f'results/{params}/dists.txt', 'w') as f:
                f.write(f'| Euclidean distance | Mahalanobis distance |\n')
                f.write(f'| --- | --- |\n')
                for edist, mdist in dists:
                    f.write(f'| {edist} | {mdist} | \n')
            plt.cla()


if __name__ == "__main__":
    main()
