#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

const int filter_size = 5;
const double eta = 0.01;
const int batch_size = 200;

vector<vector<int>> data_train(60000, vector<int>(784, 0));
vector<vector<int>> data_test(10000, vector<int>(784, 0));
vector<int> label_train(60000, 0);
vector<int> label_test(10000, 0);

vector<vector<vector<double>>> conv_w(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> conv_b(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> conv_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> sig_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> max_pooling(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> max_layer(8, vector<vector<double>>(14, vector<double>(14, 0)));

vector<double> dense_input(1568, 0);
vector<vector<double>> dense_w(1568, vector<double>(120, 0));
vector<double> dense_b(120, 0);
vector<double> dense_sum(120, 0);
vector<double> dense_sigmoid(120, 0);
vector<vector<double>> dense_w2(120, vector<double>(10, 0));
vector<double> dense_b2(10, 0);
vector<double> dense_sum2(10, 0);
vector<double> dense_softmax(10, 0);

vector<vector<double>> dw2(120, vector<double>(10, 0));
vector<double> db2(10, 0);
vector<vector<double>> dw1(1568, vector<double>(120, 0));
vector<double> db1(120, 0);

vector<vector<vector<double>>> dw_max(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> dw_conv(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> db_conv(8, vector<vector<double>>(28, vector<double>(28, 0)));

/* ************************************************************ */
/* Helper functions */
double sigmoid(double x) 
{
	if (x > 500) x = 500;
	if (x < -500) x = -500;
	return 1 / (1 + exp(-x));
}
double d_sigmoid(double x) 
{
	double sig = sigmoid(x);
	return sig * (1 - sig);
}
double softmax_den(vector<double> x, int len) 
{
	double val = 0;
	for (int i = 0; i < len; i++)
	{
		val += exp(x[i]);
	}
	return val;
}

void initialise_weights() 
{
	for (int i = 0; i < 8; i++) 
	{
		for (int j = 0; j < 28; j++) 
		{
			for (int k = 0; k < 28; k++) 
			{
				if (j < 5 && k < 5) 
				{
					conv_w[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
				}
				conv_b[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
			}
		}
	}

	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			dense_w[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}
	for (int i = 0; i < 120; i++) 
	{
		dense_b[i] = 2 * double(rand()) / RAND_MAX - 1;
	}

	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			dense_w2[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}
	for (int i = 0; i < 10; i++) 
	{
		dense_b2[i] = 2 * double(rand()) / RAND_MAX - 1;
	}
}
/* ************************************************************ */

/* ************************************************************ */
/* Forward Pass */
void forward_pass(vector<vector<int>> img) 
{
	// Convolution Operation + Sigmoid Activation
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				max_pooling[filter_dim][i][j] = 0;
				conv_layer[filter_dim][i][j] = 0;
				sig_layer[filter_dim][i][j] = 0;
				for (int k = 0; k < filter_size; k++) 
				{
					for (int l = 0; l < filter_size; l++) 
					{
						conv_layer[filter_dim][i][j] = img[i + k][j + l] * conv_w[filter_dim][k][l];
					}
				}
				sig_layer[filter_dim][i][j] = sigmoid(conv_layer[filter_dim][i][j] + conv_b[filter_dim][i][j]);
			}
		}
	}
	
	for (int i = 0; i < 8; i++)
        {
            for (int j = 0; j < 28; j++)
            {
                for (int k = 0; k < 28; k++)
                {
                    cout << setprecision(4) << conv_layer[i][j][k] << " ";
                }
                cout << endl;
            }
            cout << endl;
        }
        cout << endl;
	// MAX Pooling (max_pooling, max_layer)
	double cur_max = 0;
	int max_i = 0, max_j = 0;
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i += 2) 
		{
			for (int j = 0; j < 28; j += 2) 
			{
				max_i = i;
				max_j = j;
				cur_max = sig_layer[filter_dim][i][j];
				for (int k = 0; k < 2; k++) 
				{
					for (int l = 0; l < 2; l++) 
					{
						if (sig_layer[filter_dim][i + k][j + l] > cur_max) {
							max_i = i + k;
							max_j = j + l;
							cur_max = sig_layer[filter_dim][max_i][max_j];
						}
					}
				}
				max_pooling[filter_dim][max_i][max_j] = 1;
				max_layer[filter_dim][i / 2][j / 2] = cur_max;
			}
		}
	}

	int k = 0;
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 14; i++) 
		{
			for (int j = 0; j < 14; j++) 
			{
				dense_input[k] = max_layer[filter_dim][i][j];
				k++;
			}
		}
	}
	// Dense Layer
	for (int i = 0; i < 120; i++) 
	{
		dense_sum[i] = 0;
		dense_sigmoid[i] = 0;
		for (int j = 0; j < 1568; j++) 
		{
			dense_sum[i] += dense_w[j][i] * dense_input[j];
		}
		dense_sum[i] += dense_b[i];
		dense_sigmoid[i] = sigmoid(dense_sum[i]);
	}

	// Dense Layer 2
	for (int i = 0; i < 10; i++) 
	{
		dense_sum2[i] = 0;
		for (int j = 0; j < 120; j++) 
		{
			dense_sum2[i] += dense_w2[j][i] * dense_sigmoid[j];
		}
		dense_sum2[i] += dense_b2[i];
	}

	// Softmax Output
	double den = softmax_den(dense_sum2, 10);
	for (int i = 0; i < 10; i++) 
	{
		dense_softmax[i] = exp(dense_sum2[i]) / den;
	}
}

void update_weights() {
	for (int i = 0; i < 120; i++) 
	{
		dense_b[i] -= eta * db1[i];
		for (int j = 0; j < 10; j++) 
		{
			dense_b2[j] -= eta * db2[j];
			dense_w2[i][j] -= eta * dw2[i][j];
		}
		for (int k = 0; k < 1568; k++) 
		{
			dense_w[k][i] -= eta * dw1[k][i];
		}
	}

	for (int i = 0; i < 8; i++) 
	{
		for (int k = 0; k < 5; k++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				conv_w[i][k][j] -= eta * dw_conv[i][k][j];
			}
		}
		for (int l = 0; l < 28; l++) 
		{
			for (int m = 0; m < 28; m++) 
			{
				conv_b[i][l][m] -= eta * db_conv[i][l][m];
			}
		}
	}
}
/* ************************************************************ */

/* ************************************************************ */
/* Backward Pass */
void backward_pass(vector<double> y_hat, vector<int> y, vector<vector<int>> img) 
{
	double delta4[10];
	for (int i = 0; i < 10; i++) 
	{
		delta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
		db2[i] = delta4[i]; // Bias Changes
	}

	// Calculate Weight Changes for Dense Layer 2
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			dw2[i][j] = dense_sigmoid[i] * delta4[j];
		}
	}

	// Delta 3
	double delta3[120];
	for (int i = 0; i < 120; i++) 
	{
		delta3[i] = 0;
		for (int j = 0; j < 10; j++) 
		{
			delta3[i] += dense_w2[i][j] * delta4[j];
		}
		delta3[i] *= d_sigmoid(dense_sum[i]);
	}
	for (int i = 0; i < 120; i++) db1[i] = delta3[i]; // Bias Weight change

	// Calculate Weight Changes for Dense Layer 1
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			dw1[i][j] = dense_input[i] * delta3[j];
		}
	}

	// Delta2
	double delta2[1568];
	for (int i = 0; i < 1568; i++) 
	{
		delta2[i] = 0;
		for (int j = 0; j < 120; j++) 
		{
			delta2[i] += dense_w[i][j] * delta3[j];
		}
		delta2[i] *= d_sigmoid(dense_input[i]);
	}

	// Calc back-propagated max layer dw_max
	int k = 0;
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i += 2) 
		{
			for (int j = 0; j < 28; j += 2) 
			{
				for (int l = 0; l < 2; l++) 
				{
					for (int m = 0; m < 2; m++) 
					{
						if (max_pooling[filter_dim][i + l][j + m] == 1) 
						{
							dw_max[filter_dim][i][j] = delta2[k];
						}
						else
						{
							dw_max[filter_dim][i][j] = 0;
						}
					}
				}
				k++;
			}
		}
	}
	// Calc Conv Bias Changes
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				db_conv[filter_dim][i][j] = dw_max[filter_dim][i][j];
			}
		}
	}

	// Set Conv Layer Weight changes to 0
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 5; i++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				dw_conv[filter_dim][i][j] = 0;
			}
		}
	}

	// Calculate Weight Changes for Conv Layer
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				double cur_val = dw_max[filter_dim][i][j];
				for (int k = 0; k < 5; k++) 
				{
					for (int l = 0; l < 5; l++) 
					{
						dw_conv[filter_dim][k][l] += img[i + k][j + l] * cur_val;
					}
				}
			}
		}
	}


}
/* ************************************************************ */

void read_train_data()
{
	ifstream csvread;
	csvread.open("../mnist_train.csv", ios::in);
	if (csvread) 
	{
		//Datei bis Ende einlesen und bei ';' strings trennen
		string s;
		int data_pt = 0;
		while (getline(csvread, s)) 
		{
			stringstream ss(s);
			int pxl = 0;
			while (ss.good()) 
			{
				string substr;
				getline(ss, substr, ',');
				if (pxl == 0) 
				{
					label_train[data_pt] = stoi(substr);
				}
				else 
				{
					data_train[data_pt][pxl - 1] = stoi(substr);
				}
				pxl++;
			}
			data_pt++;
		}
		csvread.close();
	}
	else 
	{
		//cerr << "Fehler beim Lesen!" << endl;
		cerr << "Can not read data!" << endl;
	}
}

void read_test_data() 
{
	ifstream csvread;
	csvread.open("../mnist_test.csv", ios::in);
	if (csvread) 
	{
		//Datei bis Ende einlesen und bei ';' strings trennen
		string s;
		int data_pt = 0;
		while (getline(csvread, s)) 
		{
			stringstream ss(s);
			int pxl = 0;
			while (ss.good()) 
			{
				string substr;
				getline(ss, substr, ',');
				if (pxl == 0) 
				{
					label_test[data_pt] = stoi(substr);
				}
				else 
				{
					data_test[data_pt][pxl - 1] = stoi(substr);
				}
				pxl++;
			}
			data_pt++;
		}
		csvread.close();
	}
	else 
	{
		//cerr << "Fehler beim Lesen!" << endl;
		cerr << "Can not read data!" << endl;
	}
}

void give_img(vector<int> vec, vector<vector<int>>& img) 
{
	int k = 0;
	for (int i = 0; i < 32; i++) 
	{
		for (int j = 0; j < 32; j++) 
		{
			if (i < 2 || j < 2 || i > 29 || j > 29) 
			{
				img[i][j] = 0;
			}
			else 
			{
				img[i][j] = vec[k];
				k++;
			}
		}
	}
}

void give_y(int y, vector<int>& vector_y) 
{
	for (int i = 0; i < 10; i++) vector_y[i] = 0;
	vector_y[y] = 1;
}

int give_prediction() 
{
	double max_val = dense_softmax[0];
	int max_pos = 0;
	for (int i = 1; i < 10; i++) 
	{
		if (dense_softmax[i] > max_val)
		{
			max_val = dense_softmax[i];
			max_pos = i;
		}
	}

	return max_pos;
}

void write_weight_bais()
{
	ofstream conv_w_txt;
	ofstream conv_b_txt;
	ofstream dense_w_txt;
	ofstream dense_b_txt;
	ofstream dense_w2_txt;
	ofstream dense_b2_txt;
	conv_w_txt.open("conv_w.txt");
	if (!conv_w_txt.is_open()) 
	{
        cout << "Failed to open file conv_w.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 5; i++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				conv_w_txt << conv_w[filter_dim][i][j] << " ";
			}
			conv_w_txt << "\n";
		}
		conv_w_txt << "\n";
	}
	conv_w_txt.close();

	conv_b_txt.open("conv_b.txt");
	if (!conv_b_txt.is_open()) 
	{
        cout << "Failed to open file conv_b.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				conv_b_txt << conv_b[filter_dim][i][j] << " ";
			}
			conv_b_txt << "\n";
		}
		conv_b_txt << "\n";
	}
	conv_b_txt.close();

	dense_w_txt.open("dense_w.txt");
	if (!dense_w_txt.is_open()) 
	{
        cout << "Failed to open file dense_w.\n";
    }
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			dense_w_txt << dense_w[i][j] << " ";
		}
		dense_w_txt << "\n\n";
	}
	dense_w_txt.close();

	dense_b_txt.open("dense_b.txt");
	if (!dense_b_txt.is_open()) 
	{
        cout << "Failed to open file dense_b.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		dense_b_txt << dense_b[i] << "\n";
	}
	dense_b_txt.close();

	dense_w2_txt.open("dense_w2.txt");
	if (!dense_w2_txt.is_open()) 
	{
        cout << "Failed to open file dense_w2.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			dense_w2_txt << dense_w2[i][j] << " ";
		}
		dense_w2_txt << "\n\n";
	}
	dense_w2_txt.close();

	dense_b2_txt.open("dense_b2.txt");
	if (!dense_b2_txt.is_open()) 
	{
        cout << "Failed to open file dense_b2.\n";
    }
	for (int i = 0; i < 10; i++) 
	{
		dense_b2_txt << dense_b2[i] << "\n";
	}
	dense_b2_txt.close();
}

int main(int argc, char *argv[]) 
{
	time_t start, end;
	read_test_data();
	read_train_data();
	initialise_weights();
	int epoch = stoi(argv[1]);
	int num = 0;
	cout << "Start Training." << endl;
	auto startTime = chrono::high_resolution_clock::now();
	for (int i = 0; i < epoch; i++) 
	{
		for (int j = 0; j < batch_size; j++) 
		{
			cout << "\rEpoch: " << i << " --------- " << setw(3) << j << " / " << batch_size << " done." << flush;
			num = rand() % 60000;
			vector<vector<int>> img(32, vector<int>(32, 0));
			vector<int> vector_y(10, 0);
			give_y(label_train[num], vector_y);
			give_img(data_train[num], img);
			forward_pass(img);
			backward_pass(dense_softmax, vector_y, img);
			update_weights();
			//num++;
		}
	}
	cout << endl;

	write_weight_bais();
	auto endTime = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> elapsed = endTime - startTime;

	double elapsedSeconds = elapsed.count() / 1000.0;
	int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
	int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
	int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
	cout << fixed << setprecision(10);
	cout << "Training time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
	cout << endl;
	int val_len = 600;
	int cor = 0;
	int confusion_mat[10][10];
	for (int i = 0; i < 10; i++) 
	{
		for (int j = 0; j < 10; j++) confusion_mat[i][j] = 0;
	}

	cout << "Start Testing." << endl;
	for (int i = 0; i < val_len; i++) 
	{
		vector<vector<int>> img(32, vector<int>(32, 0));
		give_img(data_test[i], img);
		forward_pass(img);
		int pre = give_prediction();
		confusion_mat[label_test[i]][pre]++;
		if (pre == label_test[i]) cor++;
	}
	float accu = double(cor) / val_len;
	cout << "Accuracy: " << accu << endl;

	cout << "   0 1 2 3 4 5 6 7 8 9" << endl;
	for (int i = 0; i < 10; i++) 
	{
		cout << i << ": ";
		for (int j = 0; j < 10; j++) 
		{
			cout << confusion_mat[i][j] << " ";
		}
		cout << endl;
	}
	
	return 0;
}
