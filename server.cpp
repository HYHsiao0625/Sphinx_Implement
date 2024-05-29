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

int main()
{
    int sock_fd, new_fd;
    socklen_t addrlen;
    struct sockaddr_in my_addr, client_addr;
    int status;
    char indata[1024] = {0}, outdata[1024] = {0};
	double in_data = 0;
    double a = 0, b = 0;
    int on = 1;

    /* ************************************************************ */

    const int filter_size = 5;
    const double eta = 0.01;
    const int batch_size = 200;

    /* ************************************************************ */

    vector<vector<vector<double>>> enc_conv_w(8, vector<vector<double>>(5, vector<double>(5, 0)));
    vector<vector<vector<double>>> enc_conv_b(8, vector<vector<double>>(28, vector<double>(28, 0)));
    vector<vector<vector<double>>> enc_conv_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
    vector<vector<vector<double>>> enc_enc_sig_layer(8, vector<vector<double>>(28, vector<double>(28, 0)));
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

    while (1) 
	{
        new_fd = accept(sock_fd, (struct sockaddr *)&client_addr, &addrlen);
        printf("connected by %s:%d\n", inet_ntoa(client_addr.sin_addr),
            ntohs(client_addr.sin_port));

        while (1) 
		{   
            int data_size = 32 * 32;
            int k = 0;
            vector<vector<int>> enc_img(32, vector<int>(32, 0));
            auto startTime = chrono::high_resolution_clock::now();
            do
            {
                int nbytes = recv(new_fd, indata, sizeof(indata), 0);
                if (nbytes <= 0) 
                {
                    close(new_fd);
                    printf("client closed connection.\n");
                    break;
                }
                enc_img[k / 32][k % 32] = atoi(indata);
                data_size--;
                k++;
                memset(indata, 0, sizeof(indata));
            } while (data_size > 0);

            data_size = 8 * 5 * 5;
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
                enc_conv_w[k / 25][(k / 5) % 5][k % 5] = atof(indata);
                data_size--;
                k++;
                memset(indata, 0, sizeof(indata));
            } while (data_size > 0);
            
            data_size = 8 * 28 * 28;
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
                enc_conv_b[k / 784][(k / 28) % 28][k % 28] = atof(indata);
                data_size--;
                k++;
                memset(indata, 0, sizeof(indata));
            } while (data_size > 0);

            // Convolution Operation
            for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
            {
                for (int i = 0; i < 28; i++) 
                {
                    for (int j = 0; j < 28; j++) 
                    {
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
            
            data_size = 8 * 28 * 28;
		    k = 0;
            do
            {
                sprintf(outdata, "%f", enc_conv_layer[k / 784][(k / 28) % 28][k % 28]);
                send(new_fd, outdata, sizeof(outdata), 0);
                data_size--;
                k++;
                memset(outdata, 0, sizeof(outdata));
            } while (data_size > 0);

            data_size = 8 * 28 * 28;
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
                enc_enc_sig_layer[k / 784][(k / 28) % 28][k % 28] = atof(indata);
                data_size--;
                k++;
                memset(indata, 0, sizeof(indata));
            } while (data_size > 0);

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

            int k = 0;
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
            
            auto endTime = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> elapsed = endTime - startTime;

            double elapsedSeconds = elapsed.count() / 1000.0;
            int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
            int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
            int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
            cout << fixed << setprecision(10);
            cout << "Transmission time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
            cout << endl;
            
            int nbytes = recv(new_fd, indata, sizeof(indata), 0);
            if (nbytes <= 0) 
            {
                close(new_fd);
                printf("client closed connection.\n");
                break;
            }

            // Clear outdata for the next message
            memset(indata, 0, sizeof(indata));
            memset(outdata, 0, sizeof(outdata));
        }
    }
    close(sock_fd);

    return 0;
}