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

const char* host = "0.0.0.0";
int port = 7000;

/* ************************************************************ */
/* Training Parameter */
const int filter_size = 5;
const double eta = 0.01;
const int batch_size = 100;

/* ************************************************************ */
/* Encryption Weight */
vector<vector<vector<double>>> enc_conv_w(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> enc_conv_b(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_conv_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_sig_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_max_pooling(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_max_layer(8, vector<vector<double>>(14, vector<double>(14, 0)));

vector<double> enc_dense_input(1568, 0);
vector<vector<double>> enc_dense_w(1568, vector<double>(120, 0));
vector<double> enc_dense_b(120, 0);
vector<double> enc_dense_sum(120, 0);
vector<double> enc_dense_sigmoid(120, 0);
vector<vector<double>> enc_dense_w2(120, vector<double>(10, 0));
vector<double> enc_dense_b2(10, 0);
vector<double> enc_dense_sum2(10, 0);
vector<double> enc_dense_softmax(10, 0);

vector<vector<double>> enc_dw2(120, vector<double>(10, 0));
vector<double> enc_db2(10, 0);
vector<vector<double>> enc_dw1(1568, vector<double>(120, 0));
vector<double> enc_db1(120, 0);

vector<vector<vector<double>>> enc_dw_max(8, vector<vector<double>>(28, vector<double>(28, 0)));
vector<vector<vector<double>>> enc_dw_conv(8, vector<vector<double>>(5, vector<double>(5, 0)));
vector<vector<vector<double>>> enc_db_conv(8, vector<vector<double>>(28, vector<double>(28, 0)));

/* ************************************************************ */

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
			enc_dense_w[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}

	for (int i = 0; i < 120; i++) 
	{
		enc_dense_b[i] = 2 * double(rand()) / RAND_MAX - 1;
	}

	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			enc_dense_w2[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}

	for (int i = 0; i < 10; i++) 
	{
		enc_dense_b2[i] = 2 * double(rand()) / RAND_MAX - 1;
	}
}

void write_weight_bais()
{
	ofstream conv_w_txt;
	ofstream conv_b_txt;
	ofstream dense_w_txt;
	ofstream dense_b_txt;
	ofstream dense_w2_txt;
	ofstream dense_b2_txt;
	conv_w_txt.open("init_conv_w.txt");
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
				conv_w_txt << enc_conv_w[filter_dim][i][j] << " ";
			}
			conv_w_txt << "\n";
		}
		conv_w_txt << "\n";
	}
	conv_w_txt.close();

	conv_b_txt.open("init_conv_b.txt");
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
				conv_b_txt << enc_conv_b[filter_dim][i][j] << " ";
			}
			conv_b_txt << "\n";
		}
		conv_b_txt << "\n";
	}
	conv_b_txt.close();

	dense_w_txt.open("init_dense_w.txt");
	if (!dense_w_txt.is_open()) 
	{
        cout << "Failed to open file dense_w.\n";
    }
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			dense_w_txt << enc_dense_w[i][j] << " ";
		}
		dense_w_txt << "\n\n";
	}
	dense_w_txt.close();

	dense_b_txt.open("init_dense_b.txt");
	if (!dense_b_txt.is_open()) 
	{
        cout << "Failed to open file dense_b.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		dense_b_txt << enc_dense_b[i] << "\n";
	}
	dense_b_txt.close();

	dense_w2_txt.open("init_dense_w2.txt");
	if (!dense_w2_txt.is_open()) 
	{
        cout << "Failed to open file dense_w2.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			dense_w2_txt << enc_dense_w2[i][j] << " ";
		}
		dense_w2_txt << "\n\n";
	}
	dense_w2_txt.close();

	dense_b2_txt.open("init_dense_b2.txt");
	if (!dense_b2_txt.is_open()) 
	{
        cout << "Failed to open file dense_b2.\n";
    }
	for (int i = 0; i < 10; i++) 
	{
		dense_b2_txt << enc_dense_b2[i] << "\n";
	}
	dense_b2_txt.close();
}
/* ************************************************************ */

int main()
{
    int sock_fd, new_fd;
    socklen_t addrlen;
    struct sockaddr_in my_addr, client_addr;
    int status;
    char indata[64] = {0}, outdata[64] = {0};
    bool data_is_correct = true;
    int on = 1;
    int data_size = 0;
    int k = 0;
    double total = 0;

    // create a socket
    sock_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (sock_fd == -1) 
	{
        perror("Socket creation error");
        exit(1);
    }

    // for "Address already in use" error message
    if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(int)) == -1) 
	{
        perror("Setsockopt error");
        exit(1);
    }

    // server address
    my_addr.sin_family = AF_INET;
    inet_aton(host, &my_addr.sin_addr);
    my_addr.sin_port = htons(port);

    status = bind(sock_fd, (struct sockaddr *)&my_addr, sizeof(my_addr));
    if (status == -1) 
	{
        perror("Binding error");
        exit(1);
    }
    printf("server start at: %s:%d\n", inet_ntoa(my_addr.sin_addr), port);

    status = listen(sock_fd, 5);
    if (status == -1) 
	{
        perror("Listening error");
        exit(1);
    }
    printf("wait for connection...\n");

    addrlen = sizeof(client_addr);

   while(data_is_correct)
	{
        new_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &addrlen);
        printf("connected by %s:%d\n", inet_ntoa(client_addr.sin_addr),
            ntohs(client_addr.sin_port));

        cout << "Waiting For Training Parameter Send to Server..." << endl;
        int nbytes = recv(new_fd, indata, sizeof(indata), 0);
        if (nbytes <= 0) 
        {
            close(new_fd);
            printf("client closed connection.\n");
            break;
        }
        cout << "Train epochs is " << indata << endl;
        int epoch = atoi(indata);
        memset(indata, 0, sizeof(indata));
        cout << "Initialize Weight..." << endl;
        initialise_weights();
        write_weight_bais();
        cout << "Start Training." << endl;
        auto startTime = chrono::high_resolution_clock::now();

        for (int i = 0; i < epoch; i++) 
        {
            for (int j = 0; j < batch_size; j++) 
            {
                cout << "Epoch: " << i << " --------- " << setw(3) << j << " / " << batch_size << "." << endl;
                vector<vector<int>> enc_img(32, vector<int>(32, 0));
                vector<int> enc_vector_y(10, 0);
                /* ************************************************************ */
                /* Receiving ENC(img) */
                data_size = 32 * 32;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
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
                            close(new_fd);
                            exit(1);
                        }
                    }
                    enc_img[k / 32][k % 32] = atoi(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Receiving ENC(vector_y) */
                data_size = 10;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
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
                            close(new_fd);
                            exit(1);
                        }
                    }
                    enc_vector_y[k] = atoi(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Convolution Operation (enc_img, enc_conv_layer) */
                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 28; i++) 
                    {
                        for (int j = 0; j < 28; j++) 
                        {
                            enc_conv_layer[filter_dim][i][j] = 0;
                            for (int k = 0; k < filter_size; k++) 
                            {
                                for (int l = 0; l < filter_size; l++) 
                                {
                                    enc_conv_layer[filter_dim][i][j] = enc_img[i + k][j + l] * enc_conv_w[filter_dim][k][l];
                                }
                            }
                            enc_conv_layer[filter_dim][i][j] = enc_conv_layer[filter_dim][i][j] + enc_conv_b[filter_dim][i][j];
                        }
                    }
                }
                /* ********** TEST PASS ********** */

                /* Sending ENC(conv_layer) */
                data_size = 8 * 28 * 28;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_conv_layer[k / 784][(k / 28) % 28][k % 28]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Receiving ENC(sig_layer) */
                data_size = 8 * 28 * 28;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                    if (nbytes <= 0) 
                    {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) 
                        {
                            // 稍後再嘗試接收數據
                            continue;
                        }
                        else
                        {
                            printf("Error Receiving ENC(sig_layer): %s\n", strerror(errno));
                            close(new_fd);
                            exit(1);
                        }
                    }
                    enc_sig_layer[k / 784][(k / 28) % 28][k % 28] = atof(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* MAX Pooling (max_pooling, max_layer) */
                double cur_max = 0;
                int max_i = 0, max_j = 0;
                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 28; i += 2) 
                    {
                        for (int j = 0; j < 28; j += 2) 
                        {
                            enc_max_pooling[filter_dim][i][j] = 0;
                            max_i = i;
                            max_j = j;
                            cur_max = enc_sig_layer[filter_dim][i][j];
                            for (int k = 0; k < 2; k++) 
                            {
                                for (int l = 0; l < 2; l++) 
                                {
                                    if (enc_sig_layer[filter_dim][i + k][j + l] > cur_max) {
                                        max_i = i + k;
                                        max_j = j + l;
                                        cur_max = enc_sig_layer[filter_dim][max_i][max_j];
                                    }
                                }
                            }
                            enc_max_pooling[filter_dim][max_i][max_j] = 1;
                            enc_max_layer[filter_dim][i / 2][j / 2] = cur_max;
                        }
                    }
                }
                /* ********** TEST PASS ********** */

                /* Flat (enc_max_layer, enc_dense_input)*/
                k = 0;
                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 14; i++) 
                    {
                        for (int j = 0; j < 14; j++) 
                        {
                            enc_dense_input[k] = enc_max_layer[filter_dim][i][j];
                            k++;
                        }
                    }
                }
                /* ********** TEST PASS ********** */
                
                /* Sending ENC(dense_input) for Backward pass*/
                data_size = 1568;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_dense_input[k]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Dense Layer1 Computing*/
                for (int i = 0; i < 120; i++) 
                {
                    enc_dense_sum[i] = 0;
		            enc_dense_sigmoid[i] = 0;
                    for (int j = 0; j < 1568; j++) 
                    {
                        enc_dense_sum[i] += enc_dense_w[j][i] * enc_dense_input[j];
                    }
                    enc_dense_sum[i] += enc_dense_b[i];
                    //dense_sigmoid[i] = sigmoid(dense_sum[i]);
                }

                data_size = 120;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_dense_sum[k]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Receiving ENC(dense_sigmoid) */
                data_size = 120;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                    if (nbytes <= 0) 
                    {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) 
                        {
                            // 稍後再嘗試接收數據
                            continue;
                        }
                        else
                        {
                            printf("Error Receiving ENC(dense_sigmoid): %s\n", strerror(errno));
                            close(new_fd);
                            exit(1);
                        }
                    }
                    enc_dense_sigmoid[k] = atof(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Dense Layer2 Computing*/
                for (int i = 0; i < 10; i++) 
                {
                    enc_dense_sum2[i] = 0;
                    for (int j = 0; j < 120; j++) 
                    {
                        enc_dense_sum2[i] += enc_dense_w2[j][i] * enc_dense_sigmoid[j];
                    }
                    enc_dense_sum2[i] += enc_dense_b2[i];
                    //dense_sigmoid[i] = sigmoid(dense_sum[i]);
                }

                /* Sending ENC(dense_sum2)*/
                data_size = 10;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_dense_sum2[k]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                }while(data_size > 0);
                /* ********** TEST PASS ********** */

                /* Receiving ENC(dense_softmax) */
                data_size = 10;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                    if (nbytes <= 0) 
                    {
                        if (errno == EAGAIN || errno == EWOULDBLOCK) 
                        {
                            // 稍後再嘗試接收數據
                            continue;
                        }
                        else
                        {
                            printf("Error Receiving ENC(dense_softmax): %s\n", strerror(errno));
                            close(new_fd);
                            exit(1);
                        }
                    }
                    enc_dense_softmax[k] = atof(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                } while (data_size > 0);
                /* ********** TEST PASS ********** */
                
                total = 0;
                for (int i = 0; i < 10; i++)
                {
                    total += enc_dense_softmax[i];
                }

                cout << "total: " << total;
                cout << endl;

                if (total > 1.1)
                {
                    data_is_correct = false;
                    break;
                }

                /* Delta4 */
                double enc_delta4[10];
                for (int i = 0; i < 10; i++) 
                {
                    enc_delta4[i] = enc_dense_softmax[i] - enc_vector_y[i]; // Derivative of Softmax + Cross entropy
                    enc_db2[i] = enc_delta4[i]; // Bias Changes
                }

                for (int i = 0; i < 120; i++) 
                {
                    for (int j = 0; j < 10; j++) 
                    {
                        enc_dw2[i][j] = enc_dense_sigmoid[i] * enc_delta4[j];
                    }
                }

                /* Delta 3 */

                double enc_delta3[120];
                for (int i = 0; i < 120; i++) 
                {
                    enc_delta3[i] = 0;
                    for (int j = 0; j < 10; j++) 
                    {
                        enc_delta3[i] += enc_dense_w2[i][j] * enc_delta4[j];
                    }
                }

                /* Receiving ENC(delta3) */
                data_size = 120;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_delta3[k]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                } while (data_size > 0);

                data_size = 120;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                    if (nbytes <= 0) 
                    {
                        close(new_fd);
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
                    enc_db1[i] = enc_delta3[i];
                }

                for (int i = 0; i < 1568; i++) 
                {
                    for (int j = 0; j < 120; j++) 
                    {
                        enc_dw1[i][j] = enc_dense_input[i] * enc_delta3[j];
                    }
                }

                /* Delta2 */ 

                double enc_delta2[1568];
                for (int i = 0; i < 1568; i++) 
                {
                    enc_delta2[i] = 0;
                    for (int j = 0; j < 120; j++) 
                    {
                        enc_delta2[i] += enc_dense_w[i][j] * enc_delta3[j];
                    }
                    //enc_delta2[i] *= d_sigmoid(dense_input[i]);
                }

                data_size = 1568;
                k = 0;
                do
                {
                    sprintf(outdata, "%f", enc_delta2[k]);
                    send(new_fd, outdata, sizeof(outdata), 0);
                    data_size--;
                    k++;
                    memset(outdata, 0, sizeof(outdata));
                } while (data_size > 0);

                data_size = 1568;
                k = 0;
                do
                {
                    int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                    if (nbytes <= 0) 
                    {
                        close(new_fd);
                        printf("client closed connection.\n");
                        break;
                    }
                    enc_delta2[k] = atof(indata);
                    data_size--;
                    k++;
                    memset(indata, 0, sizeof(indata));
                } while (data_size > 0);

                k = 0;
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
                                    if (enc_max_pooling[filter_dim][i + l][j + m] == 1) 
                                    {
                                        enc_dw_max[filter_dim][i][j] = enc_delta2[k];
                                    }
                                    else
                                    {
                                        enc_dw_max[filter_dim][i][j] = 0;
                                    }
                                }
                            }
                            k++;
                        }
                    }
                }

                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 28; i++) 
                    {
                        for (int j = 0; j < 28; j++) 
                        {
                            enc_db_conv[filter_dim][i][j] = enc_dw_max[filter_dim][i][j];
                        }
                    }
                }

                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 5; i++) 
                    {
                        for (int j = 0; j < 5; j++) 
                        {
                            enc_dw_conv[filter_dim][i][j] = 0;
                        }
                    }
                }

                for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
                {
                    for (int i = 0; i < 28; i++) 
                    {
                        for (int j = 0; j < 28; j++) 
                        {
                            double cur_val = enc_dw_max[filter_dim][i][j];
                            for (int k = 0; k < 5; k++) 
                            {
                                for (int l = 0; l < 5; l++) 
                                {
                                    enc_dw_conv[filter_dim][k][l] += enc_img[i + k][j + l] * cur_val;
                                }
                            }
                        }
                    }
                }

                /* ************************************************************ */
                
                /* Update Weight*/

                for (int i = 0; i < 120; i++) 
                {
                    enc_dense_b[i] -= eta * enc_db1[i];
                    for (int j = 0; j < 10; j++) 
                    {
                        enc_dense_b2[j] -= eta * enc_db2[j];
                        enc_dense_w2[i][j] -= eta * enc_dw2[i][j];
                    }
                    for (int k = 0; k < 1568; k++) 
                    {
                        enc_dense_w[k][i] -= eta * enc_dw1[k][i];
                    }
                }

                for (int i = 0; i < 8; i++) 
                {
                    for (int k = 0; k < 5; k++) 
                    {
                        for (int j = 0; j < 5; j++) 
                        {
                            enc_conv_w[i][k][j] -= eta * enc_dw_conv[i][k][j];
                        }
                    }
                    for (int l = 0; l < 28; l++) 
                    {
                        for (int m = 0; m < 28; m++) 
                        {
                            enc_conv_b[i][l][m] -= eta * enc_db_conv[i][l][m];
                        }
                    }
                }
            }
            memset(indata, 0, sizeof(indata));
			memset(outdata, 0, sizeof(outdata));
        }
        auto endTime = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> elapsed = endTime - startTime;
        double elapsedSeconds = elapsed.count() / 1000.0;
        int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
        int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
        int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
        cout << fixed << setprecision(10);
        cout << "Training time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
        cout << endl;
        cout << "Training over."<< endl;
        cout << "Send Weight to Client..." << endl;
        data_size = 8 * 5 * 5;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_conv_w[k / 25][(k / 5) % 5][k % 5]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);

        data_size = 8 * 28 * 28;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_conv_b[k / 784][(k / 28) % 28][k % 28]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);

        data_size = 1568 * 120;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_dense_w[k / 120][k % 120]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);

        data_size = 120;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_dense_b[k]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);

        data_size = 120 * 10;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_dense_w2[k / 10][k % 10]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);
        
        data_size = 10;
        k = 0;
        do
        {
            sprintf(outdata, "%f", enc_dense_b2[k]);
            send(new_fd, outdata, sizeof(outdata), 0);
            data_size--;
            k++;
            memset(outdata, 0, sizeof(outdata));
        } while (data_size > 0);

        memset(indata, 0, sizeof(indata));
        memset(outdata, 0, sizeof(outdata));

        cout << "wait for connection..." << endl;
    }
    close(sock_fd);

    return 0;
}