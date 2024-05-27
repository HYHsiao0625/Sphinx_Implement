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
            auto endTime = chrono::high_resolution_clock::now();
            chrono::duration<double, milli> elapsed = endTime - startTime;

            double elapsedSeconds = elapsed.count() / 1000.0;
            int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
            int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
            int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
            cout << fixed << setprecision(10);
            cout << "Training time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
            cout << endl;
            for (int i = 0; i < 32; i++)
            {
                for (int j = 0; j < 32; j++)
                {
                    cout << setw(3) << enc_img[i][j];
                }
                cout << endl;
            }
            cout << "k: " << k << endl;
            int nbytes = recv(new_fd, indata, sizeof(indata), 0);
            if (nbytes <= 0) 
            {
                close(new_fd);
                printf("client closed connection.\n");
                break;
            }
            /*
            // a * x + b //
            // a = 5 //
            // b = 2 //
			in_data = atof(indata);
            memset(indata, 0, sizeof(indata));

            a = 5;
            b = 0.25;
			in_data = in_data * a + b;
            sprintf(outdata, "%.3f", in_data);
            send(new_fd, outdata, strlen(outdata), 0);
            memset(outdata, 0, sizeof(outdata));
		    */
            // Clear outdata for the next message
            memset(indata, 0, sizeof(indata));
            memset(outdata, 0, sizeof(outdata));
        }
    }
    close(sock_fd);

    return 0;
}