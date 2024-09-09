#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime> 
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>

using namespace std;
using namespace std::chrono;

const char* host = "0.0.0.0";
int port = 7000;

const int filter_size = 5;
const double eta = 0.01;
const int batch_size = 100;

/* ************************************************************ */
/* MNIST Data */
vector<vector<int>> data_train(60000, vector<int>(784, 0));
vector<vector<int>> data_test(10000, vector<int>(784, 0));
vector<int> label_train(60000, 0);
vector<int> label_test(10000, 0);

/* ************************************************************ */
/* Encryption Weight */
vector<vector<vector<double>>> enc_conv_w(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> enc_conv_b(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<double>> enc_dense_w(1568, vector<double>(120, 0));
vector<double> enc_dense_b(120, 0);
vector<vector<double>> enc_dense_w2(120, vector<double>(10, 0));
vector<double> enc_dense_b2(10, 0);

vector<vector<vector<double>>> enc_conv_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_sig_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));

vector<double> enc_dense_input(1568, 0);
vector<double> enc_dense_sum(120, 0);
vector<double> enc_dense_sigmoid(120, 0);
vector<double> enc_dense_sum2(10, 0);
vector<double> enc_dense_softmax(10, 0);

/* ************************************************************ */
/* Forward Pass*/

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

/* ************************************************************ */
/* Helper Functions */
double sigmoid(double x) 
{
	if(isnan(x))
	{
		return 0.5;
	}
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

/* ************************************************************ */

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

/* ************************************************************ */

int main() 
{ 
	int clientSocket;
    struct sockaddr_in serverAddress;
    int status;
    char indata[32] = {0}, outdata[32] = {0};
	int epochs;
	int data_size = 0;
	int k = 0;
	srand(time(NULL));
    // create a socket
    clientSocket = socket(AF_INET, SOCK_STREAM, 0); 
    if (clientSocket == -1) 
	{
        perror("Socket creation error");
        exit(1);
    }

	// specifying address 
	serverAddress.sin_family = AF_INET;
    inet_aton(host, &serverAddress.sin_addr);
    serverAddress.sin_port = htons(port);

	status = connect(clientSocket, (struct sockaddr *)&serverAddress, sizeof(serverAddress));
    if (status == -1) 
	{
        perror("Connection error");
        exit(1);
    }
	cout << "Server Connected." << endl;
	cout << "Please Enter Training Epochs:";
	cin >> epochs;

	cout << "Send Epochs to Server..." << endl;
	sprintf(outdata, "%d", epochs);
	send(clientSocket, outdata, sizeof(outdata), 0);
	memset(outdata, 0, sizeof(outdata));

	cout << "Read Train Data..." << endl;
	read_train_data();
	read_test_data();
	cout << "Start Training." << endl;
	auto startTime = chrono::high_resolution_clock::now();
	for (int i = 0; i < epochs; i++)
	{
		for (int j = 0; j < batch_size; j++)
		{
			cout << "\rEpoch: " << i << " --------- |" << setw(3) << j << " / " << batch_size << " |" << flush;
			vector<vector<int>> img(32, vector<int>(32, 0));
			vector<int> vector_y(10, 0);
			int num = rand() % 60000;
			give_y(label_train[num], vector_y);
			give_img(data_train[num], img);

			/* Enc(img) */

			/* Sending img*/
			data_size = 32 * 32;
			k = 0;
			do
			{
				sprintf(outdata, "%d", img[k / 32][k % 32]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Sending ENC(vector_y)*/
			data_size = 10;
			k = 0;
			do
			{
				sprintf(outdata, "%d", vector_y[k]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */
			
			/* Receiving ENC(conv_layer) */
			data_size = 8 * 28 * 28;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0) 
				{
					if (errno == EAGAIN || errno == EWOULDBLOCK) 
					{
						// 稍後再嘗試接收數據
						continue;
					}
					else
					{
						printf("Error Receiving ENC(conv_layer): %s\n", strerror(errno));
						close(clientSocket);
						exit(1);
					}
				}
				enc_conv_layer[k / 784][(k / 28) % 28][k % 28] = atof(indata);
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */
			
			for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
			{
				for (int i = 0; i < 28; i++) 
				{
					for (int j = 0; j < 28; j++) 
					{
						/* DEC(enc_conv_layer) */

						/* SIG(conv_layer) */
						enc_sig_layer[filter_dim][i][j] = sigmoid(enc_conv_layer[filter_dim][i][j]);
						
						/* ENC(sig_layer) */
					}
				}
			}
			/* ********** TEST PASS ********** */

			/* Sending ENC(sig_layer)*/
			data_size = 8 * 28 * 28;
			k = 0;
			do
			{
				sprintf(outdata, "%f", enc_sig_layer[k / 784][(k / 28) % 28][k % 28]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Receiving ENC(dense_input) for Backward pass */
			data_size = 1568;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0)
				{
					if (errno == EAGAIN || errno == EWOULDBLOCK) 
					{
						// 稍後再嘗試接收數據
						continue;
					}
					else
					{
						printf("Error Receiving ENC(dense_input): %s\n", strerror(errno));
						close(clientSocket);
						exit(1);
					}
				}
				enc_dense_input[k] = atof(indata);
				//cout << "\r" << enc_dense_input[k] << flush;
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Receiving ENC(dense_sum)*/
			data_size = 120;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0) 
				{
					if (errno == EAGAIN || errno == EWOULDBLOCK) 
					{
						// 稍後再嘗試接收數據
						continue;
					}
					else
					{
						printf("Error Receiving ENC(img): %s\n", strerror(errno));
						close(clientSocket);
						exit(1);
					}
				}
				enc_dense_sum[k] = atof(indata);
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			}while (data_size > 0);
			/* ********** TEST PASS ********** */

			for (int i = 0; i < 120; i++) 
			{
				//DEC(enc_dense_sum)

				//SIG(dense_sum)
				enc_dense_sigmoid[i] = sigmoid(enc_dense_sum[i]);
				//ENC(dense_sigmoid)
			}

			/* Sending ENC(dense_sigmoid)*/
			data_size = 120;
			k = 0;
			do
			{
				sprintf(outdata, "%f", enc_dense_sigmoid[k]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Receiving ENC(dense_sum2) */
			data_size = 10;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0) 
				{
					if (errno == EAGAIN || errno == EWOULDBLOCK) 
                        {
                            // 稍後再嘗試接收數據
                            continue;
                        }
                        else
                        {
                            printf("Error Receiving ENC(dense_sum2): %s\n", strerror(errno));
                            close(clientSocket);
                            exit(1);
                        }
				}
				enc_dense_sum2[k] = atof(indata);
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Softmax Output */
			double den = 0;
			den = softmax_den(enc_dense_sum2, 10);
			for (int i = 0; i < 10; i++) 
			{
				/* DEC(enc_dense_sum2) */

				/* Softmax(dense_sum2) */
				enc_dense_softmax[i] = exp(enc_dense_sum2[i]) / den;
				/* ENC(dense_softmax) */

			}

			/* Sending ENC(dense_softmax) */
			data_size = 10;
			k = 0;
			do
			{
				sprintf(outdata, "%f", enc_dense_softmax[k]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			}while(data_size > 0);
			/* ********** TEST PASS ********** */

			/* Sending ENC(delta3) */
			double enc_delta3[120];
			data_size = 120;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0) 
				{
					close(clientSocket);
					printf("client closed connection.\n");
					break;
				}
				enc_delta3[k] = atof(indata);
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			} while (data_size > 0);
			
			for (int i = 0; i < 120; i++)
			{
				/* DEC(enc_delta3) */

				/* Multiply by D_Sigmoid(dense_sum2) */
				enc_delta3[i] *= d_sigmoid(enc_dense_sum[i]);
				/* ENC(delta3) */
				
			}

			/* Sending ENC(delta3) */
			data_size = 120;
			k = 0;
			do
			{
				sprintf(outdata, "%f", enc_delta3[k]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			} while (data_size > 0);

			/* Receiving ENC(enc_delta2) */
			double enc_delta2[1568];
			data_size = 1568;
			k = 0;
			do
			{
				int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
				if (nbytes <= 0) 
				{
					close(clientSocket);
					printf("client closed connection.\n");
					break;
				}
				enc_delta2[k] = atof(indata);
				data_size--;
				k++;
				memset(indata, 0, sizeof(indata));
			} while (data_size > 0);

			for (int i = 0; i < 1568; i++)
			{
				/* DEC(enc_delta2) */

				/* Multiply by D_Sigmoid(dense_sum2) */
				enc_delta2[i] *= d_sigmoid(enc_dense_input[i]);
				/* ENC(delta2) */
				
			}

			data_size = 1568;
			k = 0;
			do
			{
				sprintf(outdata, "%f", enc_delta2[k]);
				send(clientSocket, outdata, sizeof(outdata), 0);
				data_size--;
				k++;
				memset(outdata, 0, sizeof(outdata));
			} while (data_size > 0);
			/* Clear outdata for the next message */
			memset(indata, 0, sizeof(indata));
			memset(outdata, 0, sizeof(outdata));
		}
	}
	cout << "\nTraining over."<< endl;
	cout << "Receive Weight..."<< endl;

	data_size = 8 * 5 * 5;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		conv_w[k / 25][(k / 5) % 5][k % 5] = atof(indata);
		//cout << "\r" << enc_conv_w[k / 25][(k / 5) % 5][k % 5] << flush; 
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	
	data_size = 8 * 28 * 28;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		conv_b[k / 784][(k / 28) % 28][k % 28] = atof(indata);
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	
	// Dense Layer Weight

	data_size = 1568 * 120;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		dense_w[k / 120][k % 120] = atof(indata);
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	
	data_size = 120;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		dense_b[k] = atof(indata);
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	
	data_size = 120 * 10;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		dense_w2[k / 10][k % 10] = atof(indata);
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	
	data_size = 10;
	k = 0;
	do
	{
		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) 
		{
			close(clientSocket);
			printf("client closed connection.\n");
			break;
		}
		dense_b2[k] = atof(indata);
		data_size--;
		k++;
		memset(indata, 0, sizeof(indata));
	} while (data_size > 0);
	write_weight_bais();

	int val_len = 600;
	int cor = 0;
	int confusion_mat[10][10];
	for(int i = 0; i < 10; i++)
	{
		for(int j = 0; j < 10; j++)
		{
			confusion_mat[i][j] = 0;
		}
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