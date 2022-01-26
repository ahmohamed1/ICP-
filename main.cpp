#include<iostream>
#include<vector>
#include<fstream>
#include<opencv2/core.hpp>
#include<Eigen/Dense>
#define CVPLOT_HEADER_ONLY
#include"CvPlot/core.h"

using namespace std;
#include"helper.h"

#define pi 3.14159265359


vector<Eigen::Vector2d> Apply_transform(vector<Eigen::Vector2d> P, 
										Eigen::MatrixXd rot, 
										Eigen::Vector2d t)
{
	vector<Eigen::Vector2d> temp;
	for (int i = 0; i < P.size(); i++)
	{
		Eigen::Vector2d _temp;
		_temp = rot * P[i] + t;
		temp.push_back(_temp);
	}
	return temp;
}

vector<Eigen::Vector2d> ICP_svd(vector<Eigen::Vector2d> P, 
								vector<Eigen::Vector2d> Q, 
								vector<double>& error, 
								int iterations = 10)
{
	Eigen::MatrixXd R_final = Eigen::MatrixXd::Identity(2, 2);
	Eigen::Vector2d T_final = Eigen::Vector2d::Zero();
	// Step 1: computer center data and transform them 
	Eigen::Vector2d center_of_Q = Center_data(Q);
	vector<Eigen::Vector2d> Q_centered;
	for (int i = 0; i < Q.size(); i++)
	{
		Q_centered.push_back(Q[i] - center_of_Q);
	}
	vector<Eigen::Vector2d> P_copy = P;
	// Start the loop
	for (int iter = 0; iter < iterations; iter++)
	{
		//Plot_data(P_copy, Q,50);

		vector<Eigen::Vector2d> P_centered;
		Eigen::Vector2d center_of_P = Center_data(P_copy);
		for (int i = 0; i < P.size(); i++)
		{
			P_centered.push_back(P_copy[i] - center_of_P);
		}
		// Step 2: Compute correspondences
		auto correspondences = Get_correspondence_indices(P_centered, Q_centered);

		// Step 3: compute_cross_covariance
		auto cov = compute_cross_covariance(P_centered, Q_centered, correspondences);

		// Step 4: Find  𝑅  and  𝑡  from SVD decomposition
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
		Eigen::MatrixXd R_found = Eigen::MatrixXd::Identity(2, 2);
		R_found = svd.matrixU() * svd.matrixV().transpose();
		Eigen::Vector2d T_found = center_of_Q - R_found * center_of_P;

		// Store accumulative transformations
		R_final = R_found * R_final;
		T_final += T_found;
		// Step 5: Apply transformation and computer error
		P_copy = Apply_transform(P_copy, R_found, T_found);
		double _error = Compute_Square_difference(P_copy, Q);
		error.push_back(_error);
		cout << "Squared diff: (P_corrected - Q) = " << _error << endl;
		if (_error < 0.00001)
		{
			break;
		}
	}
	// Step 5: Apply transformation and computer error
	auto Corrected_P = Apply_transform(P, R_final, T_final);
	Plot_data(P, Q, 0);
	Plot_data(Corrected_P, Q, 0);
	cout << "R: " << R_final << endl;
	cout << "t: " << T_final << endl;
	return P_copy;
}

/// <summary>
/// This function use least square
/// </summary>
/// <param name="P"></param>
/// <param name="Q"></param>
/// <param name="error"></param>
/// <param name="iterations"></param>
/// <returns></returns>
vector<Eigen::Vector2d> ICP_least_squares(vector<Eigen::Vector2d> P, 
										  vector<Eigen::Vector2d> Q, 
										  vector<double>& error, 
										  int iterations = 10)
{
	Eigen::MatrixXd R_final = Eigen::MatrixXd::Identity(2, 2);
	Eigen::Vector2d T_final = Eigen::Vector2d::Zero();
	vector<Eigen::Vector2d> P_copy = P;
	for (int i = 0; i < iterations; i++)
	{
		Plot_data(P_copy, Q, 80);
		Eigen::Vector3d x = Eigen::Vector3d::Zero();
		Eigen::MatrixXd rot = Get_R(x(2));
		Eigen::Vector2d t;
		t(0) = x(0);
		t(1) = x(1);
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
		Eigen::Vector3d g = Eigen::Vector3d::Zero(3);
		double chi = 0.0;
		// step 1: compute correspondese 
		auto correspondences = Get_correspondence_indices(P_copy, Q);

		// step 2: compute H, g and Chi
		Prepare_system(x, P_copy, Q, correspondences, H, g, chi);
		error.push_back(chi);
		// Step 3: solve the linear equation
		Eigen::Vector3d dx = -1 * H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);

		// step 4: update x and apply transformation
		x += dx;
		x(2) = atan2(sin(x(2)), cos(x(2)));
		rot = Get_R(x(2));
		t(0) = x(0);
		t(1) = x(1);
		P_copy = Apply_transform(P_copy, rot, t);
		R_final = rot * R_final;
		T_final += t;
		if (chi < 0.0001)
		{
			break;
		}
	}
	auto PPPP = Apply_transform(P, R_final, T_final);
	Plot_data(PPPP, Q, 0);
	return P_copy;
}


void Compute_normals(vector<Eigen::Vector2d> points,
	vector<Eigen::Vector2d>& normals,
	vector<Eigen::MatrixXd>& normals_at_point,
	int step = 1)
{
	normals.push_back(Eigen::Vector2d(0, 0));
	for (int i = step; i < points.size() - step; i++)
	{
		Eigen::Vector2d prev_point = points[i - step];
		Eigen::Vector2d next_point = points[i + step];
		Eigen::Vector2d curr_point = points[i];
		double dx = next_point(0) - prev_point(0);
		double dy = next_point(1) - prev_point(1);
		vector<Eigen::Vector2d> normal;
		normal.push_back(Eigen::Vector2d(0, 0));
		normal.push_back(Eigen::Vector2d(-dy, dx));
		auto norm_value = Linalg_norm(normal);
		for (int j = 0; j < normal.size(); j++)
		{
			normal[j] = normal[j] / norm_value;
		}
		//cout << normal[1].transpose() << endl;
		normals.push_back(normal[1]);
		Eigen::MatrixXd temp_normal(2, 2);
		normals_at_point.push_back(normal[1] + curr_point);
	}
	normals.push_back(Eigen::Vector2d(0, 0));
}


Eigen::MatrixXd Compute_jacobian_PP(Eigen::Vector3d _x, Eigen::MatrixXd normal, Eigen::Vector2d point)
{
	double angle = _x(2);
	double x = point(0);
	double y = point(1);
	double nx = normal(0);
	double ny = normal(1);
	Eigen::MatrixXd J(1, 3);
	J << nx, ny,
		nx* (-x * sin(angle) - y * cos(angle)) + ny * (x * cos(angle) - y * sin(angle));
	return J;
}


void Prepare_system_normal(Eigen::Vector3d x,
	vector<Eigen::Vector2d> P,
	vector<Eigen::Vector2d> Q,
	vector<Eigen::Vector2d> correspondences,
	vector<Eigen::Vector2d> Q_normals,
	Eigen::MatrixXd& H,
	Eigen::Vector3d& g,
	double& chi)
{
	for (int i = 0; i < correspondences.size(); i++)
	{
		auto p_point = P[correspondences[i](0)];
		auto q_point = Q[correspondences[i](1)];
		Eigen::MatrixXd normal = Q_normals[correspondences[i](1)];
		auto e = normal.transpose() * Error(x, p_point, q_point);
		//auto J = normal.transpose() * Jacobian(x, p_point);
		auto J = Compute_jacobian_PP(x, normal, p_point);
		double weight = 0.0;
		if (sqrt(e(0) * e(0)) < 10)
			weight = 1.0;
		cout << "e: " << e << "\n";
		//cout << "J: " << J << "\n";
		H += weight * J.transpose() * J;
		g += weight * J.transpose() * e;
		chi += e.transpose() * e;
	}
}

vector<Eigen::Vector2d> ICP_normal(vector<Eigen::Vector2d> P, vector<Eigen::Vector2d> Q, vector<double>& error, int iterations = 10)
{
	vector<Eigen::Vector2d> Q_normals;
	vector<Eigen::MatrixXd> Q_normals_at_point;
	Compute_normals(Q, Q_normals, Q_normals_at_point);
	Eigen::MatrixXd R_final = Eigen::MatrixXd::Identity(2, 2);
	Eigen::Vector2d T_final = Eigen::Vector2d::Zero();
	vector<Eigen::Vector2d> P_copy = P;
	for (int i = 0; i < iterations; i++)
	{
		Plot_data(P_copy, Q, 70);
		Eigen::Vector3d x = Eigen::Vector3d::Zero();
		Eigen::MatrixXd rot = Get_R(x(2));
		Eigen::Vector2d t;
		t(0) = x(0);
		t(1) = x(1);
		Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
		Eigen::Vector3d g = Eigen::Vector3d::Zero(3);
		double chi = 0.0;
		// step 1: compute correspondese 
		auto correspondences = Get_correspondence_indices(P_copy, Q);

		// step 2: compute H, g and Chi
		Prepare_system_normal(x, P_copy, Q, correspondences, Q_normals, H, g, chi);
		error.push_back(chi);
		// Step 3: solve the linear equation
		Eigen::Vector3d dx = -1 * H.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(g);

		// step 4: update x and apply transformation
		x += dx;
		x(2) = atan2(sin(x(2)), cos(x(2)));
		rot = Get_R(x(2));
		t(0) = x(0);
		t(1) = x(1);
		P_copy = Apply_transform(P_copy, rot, t);

		R_final = rot * R_final;
		T_final += t;

		if (chi < 0.0001)
		{
			break;
		}
	}
	//auto PPPP = Apply_transform(P, R_final, T_final);
	//Plot_data(PPPP, Q, 0);
	return P_copy;
}



int main()
{
	//Step 0: Generate data points
	double angle = pi / 4;
	Eigen::MatrixXd R_true(2, 2);
	R_true << cos(angle), -sin(angle),
		sin(angle), cos(angle);
	Eigen::Vector2d T_true{ -2, 5 };
	int num_points = 30;
	vector<Eigen::Vector2d> true_data, moved_data;
	ifstream infile("points.csv");
	int a, c;
	char b;
	while (infile >>a >>b >>c)
	{
		Eigen::Vector2d point;
		point << a, c;
		true_data.push_back(point);
	}

	//for (int i = 0; i < num_points; i++)
	//{
	//	Eigen::Vector2d point;
	//	point << i, 0.2 * i * sin(0.5 * i);
	//	true_data.push_back(point);
	//}

	for (int i = 0; i < true_data.size(); i++)
	{
		moved_data.push_back(R_true * true_data[i] + T_true);
	}

	Plot_data(true_data, moved_data, 0);
	// Add outliers
	//moved_data[10](0) = -10;
	//moved_data[10](1) = 30;
	//moved_data[20](0) = 0;
	//moved_data[20](1) = 40;
	vector<Eigen::Vector2d> Q = true_data;
	vector<Eigen::Vector2d> P = moved_data;
	vector<double> error;

	//vector<Eigen::Vector2d> Q_normals;
	//vector<Eigen::MatrixXd> Q_normals_at_point;
	//Compute_normals(Q, Q_normals, Q_normals_at_point);

	//auto P_corrected = ICP_svd(P, Q, error, 10);
	auto P_corrected = ICP_least_squares(P, Q, error, 30);
	//auto P_corrected = ICP_normal(P, Q, error);

	// Plot data
	auto axes_error = CvPlot::makePlotAxes();
	axes_error.create<CvPlot::Series>(error, "-ob");
	//for (int i = 0; i < correspondences.size(); i++)
	//{
	//	auto x1 = P_centered[correspondences[i](0)](0);
	//	auto x2 = Q_centered[correspondences[i](1)](0);
	//	auto y1 = P_centered[correspondences[i](0)](1);
	//	auto y2 = Q_centered[correspondences[i](1)](1);
	//	axes.create<CvPlot::Series>(std::vector<double>{x1,x2}, std::vector<double>{y1,y2}, "-g");
	//}

	cv::Mat mat_error = axes_error.render(420, 640);
	cv::imshow("Error", mat_error);
	cv::imwrite("CPI_SVD_ERROR.jpg", mat_error);
	cv::waitKey();

	return 0;

}




