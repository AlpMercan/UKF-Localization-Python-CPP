import numpy as np
from numpy.linalg import cholesky, inv, eig, LinAlgError
import matplotlib.pyplot as plt


class UKF:
    def __init__(self, initial_state, dt, process_noise, measurement_noise):
        self.state = np.array(initial_state)
        self.dt = dt
        self.n = len(initial_state)

        # UKF parameters Do not fucking touch these unless you have a deathwish or einstein himself
        self.alpha = 0.001
        self.beta = 2
        self.kappa = 0
        self.lambda_ = self.alpha**2 * (self.n + self.kappa) - self.n
        self.gamma = np.sqrt(self.n + self.lambda_)

        self.P = np.eye(self.n)  # initial state covariance
        self.Q = process_noise
        self.R = measurement_noise

    def ensure_positive_definite(self, matrix):
        # I did not thought this was necessary theorytically but god knows why
        eigenvalues, eigenvectors = eig(matrix)
        eigenvalues[
            eigenvalues < 0
        ] = 1e-4  # I have no idea why but if ı do not add this line it gives error
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def generate_sigma_points(self):
        sigma_points = np.zeros((2 * self.n + 1, self.n))
        sigma_points[0] = self.state

        self.P = self.ensure_positive_definite(self.P + self.Q)
        try:
            P_sqrt = cholesky(
                self.P + 1e-4 * np.eye(self.n)
            ).T  # I have no idea why but if ı do not add this line it gives error
        except LinAlgError:
            P_sqrt = np.linalg.qr(self.P)[
                0
            ].T  # Fallback to QR decomposition if Cholesky fails ı added this after dozens of tries and erros

        for i in range(self.n):
            sigma_points[i + 1] = self.state + self.gamma * P_sqrt[i]
            sigma_points[i + 1 + self.n] = self.state - self.gamma * P_sqrt[i]

        return sigma_points

    def motion_model(self, sigma_points, v, w):
        next_sigma_points = np.zeros_like(sigma_points)
        for i, point in enumerate(sigma_points):
            x, y, theta = point
            theta += w * self.dt
            x += v * self.dt * np.cos(theta)
            y += v * self.dt * np.sin(theta)
            next_sigma_points[i] = [x, y, theta]
        return next_sigma_points

    def predict(self, v, w):
        sigma_points = self.generate_sigma_points()
        sigma_points = self.motion_model(sigma_points, v, w)

        self.state = np.mean(sigma_points, axis=0)

        self.P = np.zeros((self.n, self.n))
        for i in range(2 * self.n + 1):
            diff = sigma_points[i] - self.state
            self.P += self.lambda_ / (self.n + self.lambda_) * np.outer(diff, diff)
        self.P += self.Q

    def update(self, measurement):
        predicted_measurement = self.state[:2]

        y = measurement - predicted_measurement
        S = self.R + self.P[:2, :2]
        K = self.P[:2, :2] @ inv(S)
        self.state[:2] += K @ y
        self.P[:2, :2] -= K @ S @ K.T

    def get_state(self):
        return self.state


# Example usage
initial_state = [0, 0, 0]  # x, y, theta
dt = 0.1
process_noise = np.diag([0.1, 0.1, 0.01])  # variances for x, y, theta
measurement_noise = np.diag([0.2, 0.2])  # variances for x, y measurements

ukf = UKF(initial_state, dt, process_noise, measurement_noise)

num_steps = 100  # incerease it for better options
true_states = []
estimated_states = []

for _ in range(num_steps):
    # the robots example movement m/s and rad/s respectively
    v, w = 1.0, 0.1

    # Prediction step
    ukf.predict(v, w)

    # Simulated measurement (add noise for realism)
    measurement = np.array(
        [
            ukf.state[0] + np.random.normal(0, 0.2),
            ukf.state[1] + np.random.normal(0, 0.2),
        ]
    )

    ukf.update(measurement)

    true_states.append([ukf.state[0], ukf.state[1], ukf.state[2]])
    estimated_states.append(ukf.get_state())


true_states = np.array(true_states)
estimated_states = np.array(estimated_states)

plt.figure(figsize=(12, 6))
plt.plot(
    true_states[:, 0], true_states[:, 1], "b.-", label="True Path", alpha=0.7
)  # Blue line with dots if it is solid line it overlaps
plt.plot(
    estimated_states[:, 0], estimated_states[:, 1], "r-", label="Estimated Path"
)  # Solid red line
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("UKF State Estimation vs. True State")
plt.legend()

plt.xlim(min(true_states[:, 0]) - 1, max(true_states[:, 0]) + 1)
plt.ylim(min(true_states[:, 1]) - 1, max(true_states[:, 1]) + 1)

plt.grid(True)

plt.show()
