#pragma once
double Ecladean_distance(cv::Point2d a, cv::Point2d b) {
	return sqrt((a.x * a.x - b.x * b.x) + (a.y * a.y - b.y * b.y));
}

double linalg_norm(Eigen::Vector2d p) {
	return sqrt(p(0) * p(0) + p(1) * p(1));
}

double Linalg_norm(vector<Eigen::Vector2d> P) {
	double sum = 0;
	for (auto p : P)
	{
		sum += p(0) * p(0) + p(1) * p(1);
	}
	return sqrt(sum);
}

vector<cv::Point2d> Convert_eigen_to_cv(vector<Eigen::Vector2d> list)
{
	vector<cv::Point2d> temp;
	for (int i = 0; i < list.size(); i++)
	{
		cv::Point2d p(list[i](0), list[i](1));
		temp.push_back(p);
	}
	return temp;
}

vector<Eigen::Vector2d> Get_correspondence_indices(vector<Eigen::Vector2d> P, vector<Eigen::Vector2d> Q)
{
	vector<Eigen::Vector2d> correspondences;
	for (int i = 0; i < P.size(); i++)
	{
		double min_distance = 99999999.9;
		int match_index = -1;
		for (int j = 0; j < Q.size(); j++)
		{
			double distance = linalg_norm(Q[j] - P[i]);
			if (distance < min_distance)
			{
				match_index = j;
				min_distance = distance;
			}
		}
		if (match_index != -1)
		{
			Eigen::Vector2d temp;
			temp << i, match_index;
			correspondences.push_back(temp);
		}
	}
	return correspondences;
}

Eigen::Vector2d Center_data(vector<Eigen::Vector2d> P)
{
	Eigen::Vector2d temp;
	temp << 0, 0;
	for (int i = 0; i < P.size(); i++)
	{
		temp += P[i];
	}
	temp = temp / P.size();
	return temp;
}

Eigen::MatrixXd compute_cross_covariance(vector<Eigen::Vector2d> P, vector<Eigen::Vector2d> Q, vector<Eigen::Vector2d> correspondences)
{
	Eigen::MatrixXd cov = Eigen::MatrixXd::Zero(2, 2);
	for (int i = 0; i < correspondences.size(); i++)
	{
		auto p_point = P[correspondences[i](0)];
		auto q_point = Q[correspondences[i](1)];
		double weight = 1.0;
		cov += weight * q_point * p_point.transpose();
	}
	return cov;
}

double Compute_Square_difference(vector<Eigen::Vector2d> list, vector<Eigen::Vector2d> center)
{
	vector<Eigen::Vector2d> diff;
	for (int i = 0; i < list.size(); i++)
	{
		diff.push_back(list[i] - center[i]);
	}
	return Linalg_norm(diff);
}

void Plot_data(vector<Eigen::Vector2d> P, vector<Eigen::Vector2d> Q, int sleep = 600, int Plote = 0)
{
	auto axes = CvPlot::makePlotAxes();
	axes.create<CvPlot::Series>(Convert_eigen_to_cv(P), "ob");
	axes.create<CvPlot::Series>(Convert_eigen_to_cv(Q), "or");
	cv::Mat mat = axes.render(420, 640);
	
	if (Plote != 0)
	{
		std::string savingName = std::to_string(Plote) + ".jpg";
		//std::string savingName = "datas.jpg";
		cv::imwrite(savingName, mat);
	}
	
	cv::imshow("mywindow", mat);
	cv::waitKey(sleep);
}


Eigen::MatrixXd Get_R(double angle)
{
	Eigen::MatrixXd R(2, 2);
	R << cos(angle), -sin(angle),
		sin(angle), cos(angle);
	return R;
}

Eigen::MatrixXd Get_dR(double angle)
{
	Eigen::MatrixXd R(2, 2);
	R << -sin(angle), -cos(angle),
		cos(angle), -sin(angle);
	return R;
}

Eigen::MatrixXd Jacobian(Eigen::Vector3d _x, Eigen::Vector2d point)
{
	double angle = _x(2);
	double x = point(0);
	double y = point(1);
	Eigen::MatrixXd J(2, 3);
	J << 1, 0, -sin(angle) * x - cos(angle) * y,
		0, 1, cos(angle)* x - sin(angle) * y;

	return J;
}

Eigen::Vector2d Error(Eigen::Vector3d x, Eigen::Vector2d p, Eigen::Vector2d q)
{
	auto rotation = Get_R(x(2));
	Eigen::Vector2d translation;
	translation << x(0), x(1);
	auto prediction = rotation * p - translation;
	return (prediction - q);
}

double Kernal(double threshold, Eigen::Vector2d error)
{
	if (linalg_norm(error) < threshold)
		return 1.0;
	return 0.0;
}

void Prepare_system(Eigen::Vector3d x,
	vector<Eigen::Vector2d> P,
	vector<Eigen::Vector2d> Q,
	vector<Eigen::Vector2d> correspondences,
	Eigen::MatrixXd& H,
	Eigen::Vector3d& g,
	double& chi)
{
	for (int i = 0; i < correspondences.size(); i++)
	{
		auto p_point = P[correspondences[i](0)];
		auto q_point = Q[correspondences[i](1)];
		auto e = Error(x, p_point, q_point);
		double weight = Kernal(10, e);
		auto J = Jacobian(x, p_point);
		H += weight * J.transpose() * J;
		g += weight * J.transpose() * e;
		chi += e.transpose() * e;
	}
}