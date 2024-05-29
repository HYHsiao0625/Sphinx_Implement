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
const int batch_size = 200;

/* ************************************************************ */

vector<vector<int>> data_train(60000, vector<int>(784, 0));
vector<vector<int>> data_test(10000, vector<int>(784, 0));
vector<int> label_train(60000, 0);
vector<int> label_test(10000, 0);

vector<vector<vector<double>>> enc_conv_w(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> enc_conv_b(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_conv_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_sig_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
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
					enc_conv_w[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
				}
				enc_conv_b[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
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

/* ************************************************************ */

int main() 
{ 
	int clientSocket;
    struct sockaddr_in serverAddress;
    int status;
    char indata[1024] = {0}, outdata[1024] = {0};
	double in_data = 0;
	double sig_data = 0;

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
	
	read_train_data();
	initialise_weights();
	
	vector<vector<int>> img(32, vector<int>(32, 0));
	vector<int> vector_y(10, 0);
	int num = rand() % 60000;

	give_y(label_train[num], vector_y);
	give_img(data_train[num], img);

	while (1) 
	{
		auto startTime = chrono::high_resolution_clock::now();
		int data_size = 32 * 32;
		int k = 0;
		do
		{
			sprintf(outdata, "%d", img[k / 32][k % 32]);
			send(clientSocket, outdata, sizeof(outdata), 0);
			data_size--;
			k++;
			memset(outdata, 0, sizeof(outdata));
		} while (data_size > 0);
		
		k = 0;
		data_size = 8 * 5 * 5;
		do
		{
			sprintf(outdata, "%f", enc_conv_w[k / 25][(k / 5) % 5][k % 5]);
			send(clientSocket, outdata, sizeof(outdata), 0);
			data_size--;
			k++;
			memset(outdata, 0, sizeof(outdata));
		} while (data_size > 0);

		data_size = 8 * 28 * 28;
		k = 0;
		do
		{
			sprintf(outdata, "%f", enc_conv_b[k / 784][(k / 28) % 28][k % 28]);
			send(clientSocket, outdata, sizeof(outdata), 0);
			data_size--;
			k++;
			memset(outdata, 0, sizeof(outdata));
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
			enc_conv_layer[k / 784][(k / 28) % 28][k % 28] = atof(indata);
			data_size--;
			k++;
			memset(indata, 0, sizeof(indata));
		} while (data_size > 0);

		for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
		{
			for (int i = 0; i < 28; i++) 
			{
				for (int j = 0; j < 28; j++) 
				{
					//decryption

					//sigmoid
					enc_sig_layer[filter_dim][i][j] = sigmoid(enc_conv_layer[filter_dim][i][j]);
				
					//encryption
				}
			}
		}

		data_size = 8 * 28 * 28;
		k = 0;
		do
		{
			sprintf(outdata, "%f", enc_sig_layer[k / 784][(k / 28) % 28][k % 28]);
			send(clientSocket, outdata, sizeof(outdata), 0);
			data_size--;
			k++;
			memset(outdata, 0, sizeof(outdata));
		} while (data_size > 0);
		
		auto endTime = chrono::high_resolution_clock::now();
		chrono::duration<double, milli> elapsed = endTime - startTime;
		double elapsedSeconds = elapsed.count() / 1000.0;
		int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
		int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
		int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
		cout << fixed << setprecision(10);
		cout << "Transmission time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
		cout << endl;

		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) {
			close(clientSocket);
			printf("Server closed connection.\n");
			break;
		}

		// Clear outdata for the next message
		memset(indata, 0, sizeof(indata));
		memset(outdata, 0, sizeof(outdata));
	}

	return 0; 
}
