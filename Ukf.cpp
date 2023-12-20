#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using namespace Eigen;
using namespace std;

class UKF {
public:
    UKF(VectorXd initial_state, double dt, MatrixXd process_noise, MatrixXd measurement_noise) {
        state = initial_state;
        this->dt = dt;
        n = state.size();

        // UKF parameters
        alpha = 0.001;
        beta = 2;
        kappa = 0;
        lambda = alpha * alpha * (n + kappa) - n;
        gamma = sqrt(n + lambda);

        P = MatrixXd::Identity(n, n);  // initial state covariance
        Q = process_noise;
        R = measurement_noise;
    }

    void ensure_positive_definite(MatrixXd& matrix) {
        SelfAdjointEigenSolver<MatrixXd> solver(matrix);
        matrix = solver.eigenvectors() * solver.eigenvalues().cwiseMax(1e-4).asDiagonal() * solver.eigenvectors().transpose();
    }

    MatrixXd generate_sigma_points() {
        MatrixXd sigma_points(2 * n + 1, n);
        sigma_points.row(0) = state.transpose();

        ensure_positive_definite(P);
        MatrixXd P_sqrt = (P + 1e-4 * MatrixXd::Identity(n, n)).llt().matrixL().transpose();

        for (int i = 0; i < n; ++i) {
            sigma_points.row(i + 1) = (state + gamma * P_sqrt.col(i)).transpose();
            sigma_points.row(i + 1 + n) = (state - gamma * P_sqrt.col(i)).transpose();
        }

        return sigma_points;
    }

    void motion_model(MatrixXd& sigma_points, double v, double w) {
        for (int i = 0; i < sigma_points.rows(); ++i) {
            double x = sigma_points(i, 0);
            double y = sigma_points(i, 1);
            double theta = sigma_points(i, 2);

            theta += w * dt;
            x += v * dt * cos(theta);
            y += v * dt * sin(theta);

            sigma_points(i, 0) = x;
            sigma_points(i, 1) = y;
            sigma_points(i, 2) = theta;
        }
    }

    void predict(double v, double w) {
        MatrixXd sigma_points = generate_sigma_points();
        motion_model(sigma_points, v, w);

        state = sigma_points.colwise().mean();

        P = MatrixXd::Zero(n, n);
        for (int i = 0; i < 2 * n + 1; ++i) {
            VectorXd diff = sigma_points.row(i) - state.transpose();
            P += lambda / (n + lambda) * diff * diff.transpose();
        }
        P += Q;
    }

    void update(VectorXd measurement) {
        VectorXd predicted_measurement = state.head(2);

        VectorXd y = measurement - predicted_measurement;
        MatrixXd S = R + P.topLeftCorner(2, 2);
        MatrixXd K = P.topLeftCorner(2, 2) * S.inverse();
        state.head(2) += K * y;
        P.topLeftCorner(2, 2) -= K * S * K.transpose();
    }

    VectorXd get_state() {
        return state;
    }

private:
    VectorXd state;
    double dt;
    int n;
    double alpha, beta, kappa, lambda, gamma;
    MatrixXd P, Q, R;
};

int main() {
    VectorXd initial_state(3);
    initial_state << 0, 0, 0;  // x, y, theta

    double dt = 0.1;
    MatrixXd process_noise = MatrixXd::Zero(3, 3);
    process_noise.diagonal() << 0.1, 0.1, 0.01;
    MatrixXd measurement_noise = MatrixXd::Zero(2, 2);
    measurement_noise.diagonal() << 0.2, 0.2;

    UKF ukf(initial_state, dt, process_noise, measurement_noise);

    int num_steps = 100;
    vector<VectorXd> true_states, estimated_states;

    default_random_engine generator;
    normal_distribution<double> distribution(0.0, 0.2);

    for (int i = 0; i < num_steps; ++i) {
        double v = 1.0, w = 0.1;  // Example movement

        // Prediction step
        ukf.predict(v, w);

        // Simulated measurement
        VectorXd measurement(2);
        measurement << ukf.get_state()(0) + distribution(generator),
                       ukf.get_state()(1) + distribution(generator);

        ukf.update(measurement);

        true_states.push_back(ukf.get_state());
        estimated_states.push_back(ukf.get_state());
    }

    // Output the results
    for (int i = 0; i < num_steps; ++i) {
        cout << "True State: " << true_states[i].transpose() << ", Estimated State: " << estimated_states[i].transpose() << endl;
    }

    return 0;
}
