#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cmath>

using namespace std;
const char* host = "0.0.0.0";
int port = 7000;

double sigmoid(double x) 
{
	if (x > 500) x = 500;
	if (x < -500) x = -500;
	return 1 / (1 + exp(-x));
}

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

	while (1) 
	{
		printf("Please input message: ");
		cin.getline(outdata, sizeof(outdata)); // Use getline for safer input
		printf("send: %s\n", outdata);
		send(clientSocket, outdata, strlen(outdata), 0);
		memset(outdata, 0, sizeof(outdata));

		int nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) {
			close(clientSocket);
			printf("Server closed connection.\n");
			break;
		}
		printf("recv: %s\n", indata);
		in_data = atof(indata);
		memset(indata, 0, sizeof(indata));

		sig_data = sigmoid(in_data);
		sprintf(outdata, "%.3f", sig_data);
		cout << "sigmood(in_data) : " << sig_data << endl;
		send(clientSocket, outdata, strlen(outdata), 0);
		memset(outdata, 0, sizeof(outdata));

		nbytes = recv(clientSocket, indata, sizeof(indata), 0);
		if (nbytes <= 0) {
			close(clientSocket);
			printf("Server closed connection.\n");
			break;
		}
		printf("recv: %s\n", indata);

		// Clear outdata for the next message
		memset(indata, 0, sizeof(indata));
		memset(outdata, 0, sizeof(outdata));
	}

	return 0; 
}
