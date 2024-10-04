#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <thread>
#include <mutex>

#include "seal/seal.h"
#include "../include/ckks.hpp"

using namespace std;
using namespace std::chrono;
using namespace seal;

class CNN
{
private:
    /* ************************************************************ */
	/* Training Parameter */
	int filter_size;
	double eta;
	int batch_size;
	double loss;
	int _t;

    /* MNIST Data */
    vector<vector<int>> _dataTrain;
    vector<vector<int>> _dataTest;
    vector<int> _labelTrain;
    vector<int> _labelTest;

    /* Weight */
    vector<vector<vector<double>>> _convW;
    vector<vector<vector<double>>> _convB;
	vector<vector<vector<double>>> _convLayer;
	vector<vector<vector<double>>> _sigLayer;
	vector<vector<vector<double>>> _maxPooling;
	vector<vector<vector<double>>> _maxLayer;
    vector<double> _denseInput;
	vector<vector<double>> _denseW;
	vector<double> _denseB;
	vector<double> _denseSum;
	vector<double> _denseSigmoid;
	vector<vector<double>> _denseW2;
	vector<double> _denseB2;
	vector<double> _denseSum2;
	vector<double> _denseSoftmax;

	/* Encrypt Parameters */
	CKKS _ckks;
	Ciphertext _zero, _one, _eta, _cipherTemp;
	PublicKey _publickey;
	SecretKey _secretkey;
	double _decrypted;

    /* Encrypt Data */
	vector<vector<Ciphertext>> _encImg;
	vector<int> _encLabel;

	/* Encrypt Weight */
    vector<vector<vector<double>>> _encConvW;
	vector<vector<vector<double>>> _encConvB;
	vector<vector<vector<double>>> _encConvLayer;
	vector<vector<vector<double>>> _encSigLayer;
	vector<vector<vector<double>>> _encMaxPooling;
	vector<vector<vector<double>>> _encMaxLayer;

	vector<double> _encDenseInput;
	vector<vector<double>> _encDenseW;
	vector<double> _encDenseB;
	vector<double> _encDenseSum;
	vector<double> _encDenseSigmoid;
	vector<vector<double>> _encDenseW2;
	vector<double> _encDenseB2;
	vector<double> _encDenseSum2;
	vector<double> _encDenseSoftmax;

	vector<vector<double>> _encDffW2;
	vector<double> _encDffB2;
	vector<vector<double>> _encDffW1;
	vector<double> _encDffB1;
	vector<vector<vector<double>>> _encDffMaxW;
	vector<vector<vector<double>>> _encDffConvW;
	vector<vector<vector<double>>> _encDffConvB;
public:
    CNN(/* args */);
    ~CNN();
	void Init();
	double Sigmoid(double x);
	double DffSigmoid(double x);
	double SoftmaxDen(vector<double> x, int len);
    void Train(int epochs);
    void ReadData();
    void PrintImg(vector<vector<double>> img);
    void InitialiseWeight();
	void WriteInitWeight();

	void EncryptWeight();
	void DecryptWeight();
	void GiveImg(vector<int> vec, vector<vector<int>>& img);
	void GiveLabel(int y, vector<int>& vector_y);
	int GivePrediction();
	void EncryptData(vector<vector<int>>& img, vector<int>& label);
	void EncryptForward(vector<vector<Ciphertext>>& _encImg);
	void subTaskEncryptForward(int start, int end);
	void EncryptBackword(vector<double>& y_hat, vector<int>& y, vector<vector<Ciphertext>>& _encImg);
	void UpdateWeight();
	void WriteTrainedWeight();
	void Predict();
	void Forward(vector<vector<int>>& img);

	void AdamOptimizer(double _lr, double _b1, double _b2, double _eps);

};

CNN::CNN()
{
	
}

CNN::~CNN()
{

}

void CNN::Init()
{
	/* Parameter initialization */
	/* Training Parameter */
	filter_size = 5;
	eta = 0.01;
	batch_size = 100;
	loss = 0;
	_t = 1;

	/* Train & Test Data */
	_dataTrain = vector<vector<int>>(60000, vector<int>(784, 0));
    _dataTest = vector<vector<int>> (10000, vector<int>(784, 0));
    _labelTrain = vector<int>(60000, 0);
    _labelTest = vector<int>(10000, 0);

    /* Weight */
    _convW = vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 0)));
    _convB = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
    _convLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_sigLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_maxPooling = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_maxLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	
	_denseInput = vector<double>(1568, 0);
	_denseW = vector<vector<double>>(1568, vector<double>(120, 0));
	_denseB = vector<double>(120, 0);
	_denseSum = vector<double>(120, 0);
	_denseSigmoid = vector<double>(120, 0);
	_denseW2 = vector<vector<double>>(120, vector<double>(10, 0));
	_denseB2 = vector<double>(10, 0);
	_denseSum2 = vector<double>(10, 0);
	_denseSoftmax = vector<double>(10, 0);

	/* Init Encryption Variables */
	// Initial ckks
	_ckks.initParams();
	// Generate keyset
	_ckks.generateKey(&_publickey, &_secretkey);
	// Generate cipher variables for evaluation or initialization purposes.
	_ckks.encryptPlain(0, _publickey, &_zero);
	_ckks.encryptPlain(1, _publickey, &_one);
	_ckks.encryptPlain(eta, _publickey, &_eta);

	/* Encrypt Data */
	_encImg = vector<vector<Ciphertext>>(32, vector<Ciphertext>(32, _zero));
	_encLabel = vector<int>(10, 0);

	/* Encrypt Weight */
	_encConvW = vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 0)));
	_encConvB = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_encConvLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_encSigLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_encMaxPooling = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_encMaxLayer = vector<vector<vector<double>>>(8, vector<vector<double>>(14, vector<double>(14, 0)));

	_encDenseInput = vector<double>(1568, 0);
	_encDenseW = vector<vector<double>>(1568, vector<double>(120, 0));
	_encDenseB = vector<double>(120, 0);
	_encDenseSum = vector<double>(120, 0);
	_encDenseSigmoid = vector<double>(120, 0);
	_encDenseW2 = vector<vector<double>>(120, vector<double>(10, 0));
	_encDenseB2 = vector<double>(10, 0);
	_encDenseSum2 = vector<double>(10, 0);
	_encDenseSoftmax = vector<double>(10, 0);

	_encDffW2 = vector<vector<double>>(120, vector<double>(10, 0));
	_encDffB2 = vector<double>(10, 0);
	_encDffW1 = vector<vector<double>>(1568, vector<double>(120, 0));
	_encDffB1 = vector<double>(120, 0);
	_encDffMaxW = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	_encDffConvW = vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 0)));
	_encDffConvB = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));

	InitialiseWeight();
	WriteInitWeight();
}
/* ************************************************************ */
/* Helper functions */
double CNN::Sigmoid(double x) 
{
	if (x > 500) x = 500;
	if (x < -500) x = -500;
	return 1 / (1 + exp(-x));
}

double CNN::DffSigmoid(double x) 
{
	double sig = Sigmoid(x);
	return sig * (1 - sig);
}

double CNN::SoftmaxDen(vector<double> x, int len) 
{
	double val = 0;
	for (int i = 0; i < len; i++)
	{
		val += exp(x[i]);
	}
	return val;
}

void CNN::ReadData()
{
    /* Read Train Data */
    ifstream csvread;
    csvread.open("../data/mnist_train.csv", ios::in);
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
                    _labelTrain[data_pt] = stoi(substr);
                }
                else 
                {
                    _dataTrain[data_pt][pxl - 1] = stoi(substr);
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
    
    /* Read Test Data */
    csvread.open("../data/mnist_test.csv", ios::in);
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
                    _labelTest[data_pt] = stoi(substr);
                }
                else 
                {
                    _dataTest[data_pt][pxl - 1] = stoi(substr);
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

void CNN::PrintImg(vector<vector<double>> _img)
{
    for(int i = 0; i < 14; i++)
    {
        for(int	j = 0; j < 14; j++)
		{
			cout << setw(8) << fixed << setprecision(4) << _img[i][j];
		}
		cout << endl;
    }
    cout << endl;
}
/* ************************************************************ */
void CNN::InitialiseWeight()
{
	for (int i = 0; i < 8; i++) 
	{
		for (int j = 0; j < 28; j++) 
		{
			for (int k = 0; k < 28; k++) 
			{
				if (j < 5 && k < 5) 
				{
					_convW[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
				}
				_convB[i][j][k] = 2 * double(rand()) / RAND_MAX - 1; // Random double value between -1 and 1
			}
		}
	}

	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			_denseW[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}
	for (int i = 0; i < 120; i++) 
	{
		_denseB[i] = 2 * double(rand()) / RAND_MAX - 1;
	}

	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			_denseW2[i][j] = 2 * double(rand()) / RAND_MAX - 1;
		}
	}
	for (int i = 0; i < 10; i++) 
	{
		_denseB2[i] = 2 * double(rand()) / RAND_MAX - 1;
	}
}

void CNN::WriteInitWeight()
{
	ofstream _initConvW;
	ofstream _initConvB;
	ofstream _initDenseW;
	ofstream _initDenseB;
	ofstream _initDenseW2;
	ofstream _initDenseB2;
	_initConvW.open("init_conv_w.txt");
	if (!_initConvW.is_open()) 
	{
        cout << "Failed to open file init_conv_w.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 5; i++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				_initConvW << _convW[filter_dim][i][j] << " ";
			}
			_initConvW << "\n";
		}
		_initConvW << "\n";
	}
	_initConvW.close();

	_initConvB.open("init_conv_b.txt");
	if (!_initConvB.is_open()) 
	{
        cout << "Failed to open file init_conv_b.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				_initConvB << _convB[filter_dim][i][j] << " ";
			}
			_initConvB << "\n";
		}
		_initConvB << "\n";
	}
	_initConvB.close();

	_initDenseW.open("init_dense_w.txt");
	if (!_initDenseW.is_open()) 
	{
        cout << "Failed to open file init_dense_w.\n";
    }
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			_initDenseW << _denseW[i][j] << " ";
		}
		_initDenseW << "\n\n";
	}
	_initDenseW.close();

	_initDenseB.open("init_dense_b.txt");
	if (!_initDenseB.is_open()) 
	{
        cout << "Failed to open file init_dense_b.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		_initDenseB << _denseB[i] << "\n";
	}
	_initDenseB.close();

	_initDenseW2.open("init_dense_w2.txt");
	if (!_initDenseW2.is_open()) 
	{
        cout << "Failed to open file init_dense_w2.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			_initDenseW2 << _denseW2[i][j] << " ";
		}
		_initDenseW2 << "\n\n";
	}
	_initDenseW2.close();

	_initDenseB2.open("init_dense_b2.txt");
	if (!_initDenseB2.is_open()) 
	{
        cout << "Failed to open file init_dense_b2.\n";
    }
	for (int i = 0; i < 10; i++) 
	{
		_initDenseB2 << _denseB2[i] << "\n";
	}
	_initDenseB2.close();
}

void CNN::EncryptWeight()
{
	/* Encrypt Weight */
	_encConvW = _convW;
    _encConvB = _convB;
    _encDenseW = _denseW;
    _encDenseB = _denseB ;
    _encDenseW2 = _denseW2;
    _encDenseB2 = _denseB2;
}

void CNN::DecryptWeight()
{
	/* Decrypt Weight */
	_convW = _encConvW;
    _convB = _encConvB;
    _denseW = _encDenseW;
    _denseB = _encDenseB;
    _denseW2 = _encDenseW2;
    _denseB2 = _encDenseB2;
}

void CNN::GiveImg(vector<int> vec, vector<vector<int>>& img) 
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

void CNN::GiveLabel(int y, vector<int>& label) 
{
	for (int i = 0; i < 10; i++)
	{
		label[i] = 0;
	}
	label[y] = 1;
}

int CNN::GivePrediction()
{
	double max_val = _denseSoftmax[0];
	int max_pos = 0;
	for (int i = 1; i < 10; i++) 
	{
		if (_denseSoftmax[i] > max_val)
		{
			max_val = _denseSoftmax[i];
			max_pos = i;
		}
	}

	return max_pos;
}

void CNN::EncryptData(vector<vector<int>>& img, vector<int>& label)
{
	/* Encrypt Data */
	// img

	for (auto& i: img) {
		for (auto& j: i)
			printf("%3d ", j);
		cout << endl;
	}
	
	for (size_t x = 0; x < img.size(); x++)
		for (size_t y = 0; y < img[x].size(); y++) {
			cout << "\rEncrypting Image: " << x * img.size() + y + 1 << "/" << img.size() * img[0].size() << " " << !img[x][y] << " : " << img[x][y] << flush;
			if (!img[x][y])
				_encImg[x][y] = _zero;
			else
				_ckks.encryptPlain(img[x][y], _publickey, &_encImg[x][y]);
		}
	cout << endl;
			
	// label
	_encLabel = label;
}

void CNN::subTaskEncryptForward(int start, int end) {
	Ciphertext localCipher;
	double localDecrypted;

	for (int filter_dim = start; filter_dim < end; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				_encConvLayer[filter_dim][i][j] = 0;
				_encSigLayer[filter_dim][i][j] = 0;
				_encMaxPooling[filter_dim][i][j] = 0;
				for (int k = 0; k < filter_size; k++) 
				{
					for (int l = 0; l < filter_size; l++) 
					{
						_ckks.encryptPlain(_encConvW[filter_dim][k][l], _publickey, &localCipher);
						_ckks.evaluateCipher(&localCipher, "*", &_encImg[i + k][j + l]);
						_ckks.decryptCipher(localCipher, _secretkey, &localDecrypted);
						_encConvLayer[filter_dim][i][j] = localDecrypted;
						// printf("\r%02i/%02i/%02i/%02i/%02i", l, k, j, i, filter_dim);
						// cout << flush;
					}
				}
				_encSigLayer[filter_dim][i][j] = Sigmoid(_encConvLayer[filter_dim][i][j] + _encConvB[filter_dim][i][j]);
			}
		}
		//PrintImg(_encSigLayer[filter_dim]);
	}
}

void CNN::EncryptForward(vector<vector<Ciphertext>>& _encImg)
{
	cout << "Convolution with Sigmoid.\n";

	/* Seperate convolution with sigmoid to 8 parts. */
	unsigned int threadsCount = min(8U, thread::hardware_concurrency());
	printf("Physical Threads: %i, Allocated Threads: %i\n", thread::hardware_concurrency(), threadsCount);
	
    int tasksPerThread = 8 / threadsCount;
    int remainingTasks = 8 % threadsCount;

	vector<thread> threads;

    int current_filter = 0;
    for (unsigned int t = 0; t < threadsCount; ++t) {
        int start = current_filter;
        int end = start + tasksPerThread + (t < remainingTasks ? 1 : 0);
        threads.emplace_back(&CNN::subTaskEncryptForward, this, start, end);
        current_filter = end;
    }

    for (auto& th : threads)
        th.join();
	/* --------------------------------------------- */

	cout << "Max Pooling.\n";
	
	/* MAX Pooling (max_pooling, max_layer) */ 
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
				cur_max = _encSigLayer[filter_dim][i][j];
				for (int k = 0; k < 2; k++) 
				{
					for (int l = 0; l < 2; l++) 
					{
						if (_encSigLayer[filter_dim][i + k][j + l] > cur_max) 
						{
							max_i = i + k;
							max_j = j + l;
							cur_max = _encSigLayer[filter_dim][max_i][max_j];
						}
					}
				}
				_encMaxPooling[filter_dim][max_i][max_j] = 1;
				_encMaxLayer[filter_dim][i / 2][j / 2] = cur_max;
			}
		}
	}
	
	/* Flat (enc_max_layer, enc_dense_input) */
	int k = 0;
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 14; i++) 
		{
			for (int j = 0; j < 14; j++) 
			{
				_encDenseInput[k] = _encMaxLayer[filter_dim][i][j];
				k++;
			}
		}
	}

	/* Dense Layer1 Computing */
	for (int i = 0; i < 120; i++) 
	{
		_encDenseSum[i] = 0;
		_encDenseSigmoid[i] = 0;
		for (int j = 0; j < 1568; j++) 
		{
			_encDenseSum[i] += _encDenseW[j][i] * _encDenseInput[j];
		}
		_encDenseSum[i] += _encDenseB[i];
		_encDenseSigmoid[i] = Sigmoid(_encDenseSum[i]);
	}

	/* Dense Layer2 Computing */
	for (int i = 0; i < 10; i++) 
	{
		_encDenseSum2[i] = 0;
		for (int j = 0; j < 120; j++) 
		{
			_encDenseSum2[i] += _encDenseW2[j][i] * _encDenseSigmoid[j];
		}
		_encDenseSum2[i] += _encDenseB2[i];
	}

	/* Softmax Output */ 
	double den = SoftmaxDen(_encDenseSum2, 10);
	for (int i = 0; i < 10; i++) 
	{
		_encDenseSoftmax[i] = exp(_encDenseSum2[i]) / den;
	}
}

void CNN::EncryptBackword(vector<double>& y_hat, vector<int>& y, vector<vector<Ciphertext>>& _encImg) 
{
	double _encDelta4[10];
	for (int i = 0; i < 10; i++) 
	{
		_encDelta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
		_encDffB2[i] = _encDelta4[i]; // Bias Changes
		loss += abs(_encDelta4[i]);
	}

	// Calculate Weight Changes for Dense Layer 2
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			_encDffW2[i][j] = _encDenseSigmoid[i] * _encDelta4[j];
		}
	}

	// Delta 3
	double _encDelta3[120];
	for (int i = 0; i < 120; i++) 
	{
		_encDelta3[i] = 0;
		for (int j = 0; j < 10; j++) 
		{
			_encDelta3[i] += _encDenseW2[i][j] * _encDelta4[j];
		}
		_encDelta3[i] *= DffSigmoid(_encDenseSum[i]);
	}

	for (int i = 0; i < 120; i++)
	{
		_encDffB1[i] = _encDelta3[i]; // Bias Weight change
	}

	// Calculate Weight Changes for Dense Layer 1
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			_encDffW1[i][j] = _encDenseInput[i] * _encDelta3[j];
		}
	}

	// _encDelta2
	double _encDelta2[1568];
	for (int i = 0; i < 1568; i++) 
	{
		_encDelta2[i] = 0;
		for (int j = 0; j < 120; j++) 
		{
			_encDelta2[i] += _encDenseW[i][j] * _encDelta3[j];
		}
		_encDelta2[i] *= DffSigmoid(_encDenseInput[i]);
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
						if (_encMaxPooling[filter_dim][i + l][j + m] == 1) 
						{
							_encDffMaxW[filter_dim][i][j] = _encDelta2[k];
						}
						else
						{
							_encDffMaxW[filter_dim][i][j] = 0;
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
				_encDffConvB[filter_dim][i][j] = _encDffMaxW[filter_dim][i][j];
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
				_encDffConvW[filter_dim][i][j] = 0;
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
				double cur_val = _encDffMaxW[filter_dim][i][j];
				for (int k = 0; k < 5; k++) 
				{
					for (int l = 0; l < 5; l++) 
					{	
						_ckks.decryptCipher(_encImg[i + k][j + l], _secretkey, &_decrypted);
						_encDffConvW[filter_dim][k][l] += _decrypted * cur_val;
						printf("\r%02i/%02i/%02i/%02i/%02i", l, k, j, i, filter_dim);
						cout << flush;
					}
				}
			}
		}
	}
	cout << endl;
}

void CNN::UpdateWeight()
{
	for (int i = 0; i < 120; i++) 
	{
		_encDenseB[i] -= eta * _encDffB1[i];
		for (int j = 0; j < 10; j++) 
		{
			_encDenseB2[j] -= eta * _encDffB2[j];
			_encDenseW2[i][j] -= eta * _encDffW2[i][j];
		}
		for (int k = 0; k < 1568; k++) 
		{
			_encDenseW[k][i] -= eta * _encDffW1[k][i];
		}
	}

	for (int i = 0; i < 8; i++) 
	{
		for (int k = 0; k < 5; k++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				_encConvW[i][k][j] -= eta * _encDffConvW[i][k][j];
			}
		}
		for (int l = 0; l < 28; l++) 
		{
			for (int m = 0; m < 28; m++) 
			{
				_encConvB[i][l][m] -= eta * _encDffConvB[i][l][m];
			}
		}
	}
}

void CNN::Train(int epochs)
{
	time_t start, end;
	cout << "Start Training." << endl;
	auto startTime = chrono::high_resolution_clock::now();
	int num = 0;
	double Loss = 0;
	ReadData();
	EncryptWeight();
	for (int i = 0; i < epochs; i++) 
	{
		loss = 0;
		_t = 1;
		for (int j = 0; j < batch_size; j++)
		{
			num = rand() % 60000;
			vector<vector<int>> img(32, vector<int>(32, 0));
			vector<int> label(10, 0);
			cout << "GiveLabel\n";
			GiveLabel(_labelTrain[num], label);
			cout << "GiveImg\n";
			GiveImg(_dataTrain[num], img);
			cout << "EncryptData\n";
			EncryptData(img, label);
			cout << "EncryptForward\n";
			EncryptForward(_encImg);
			cout << "EncryptBackword\n";
			EncryptBackword(_encDenseSoftmax, _encLabel, _encImg);
			//UpdateWeight();
			cout << "AdamOptimizer\n";
			AdamOptimizer(0.01, 0.9, 0.999, 1e-8);
			cout << "\rEpoch: " << i << " --------- |" << setw(3) << j << " / " << batch_size << " | Loss: " << fixed << Loss << " |" << flush;
		}
		Loss = loss / batch_size;
	}
	cout << endl;
	DecryptWeight();
	WriteTrainedWeight();
	auto endTime = chrono::high_resolution_clock::now();
	chrono::duration<double, milli> elapsed = endTime - startTime;

	double elapsedSeconds = elapsed.count() / 1000.0;
	int elapsedMinutes = static_cast<int>(elapsedSeconds / 60);
	int elapsedSeconds_ = static_cast<int>(elapsedSeconds) % 60;
	int elapsedMilliseconds = static_cast<int>(elapsedSeconds * 1000) % 1000;
	cout << fixed << setprecision(10);
	cout << "Training time : " << elapsedMinutes << "m " << elapsedSeconds_ << "s " << elapsedMilliseconds << "ms";
	cout << endl;
}

void CNN::WriteTrainedWeight()
{
	ofstream _trainConvW;
	ofstream _trainConvB;
	ofstream _trainDenseW;
	ofstream _trainDenseB;
	ofstream _trainDenseW2;
	ofstream _trainDenseB2;
	_trainConvW.open("trained_conv_w.txt");
	if (!_trainConvW.is_open()) 
	{
        cout << "Failed to open file trained_conv_w.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 5; i++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				_trainConvW << _convW[filter_dim][i][j] << " ";
			}
			_trainConvW << "\n";
		}
		_trainConvW << "\n";
	}
	_trainConvW.close();

	_trainConvB.open("trained_conv_b.txt");
	if (!_trainConvB.is_open()) 
	{
        cout << "Failed to open file trained_conv_b.\n";
    }
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				_trainConvB << _convB[filter_dim][i][j] << " ";
			}
			_trainConvB << "\n";
		}
		_trainConvB << "\n";
	}
	_trainConvB.close();

	_trainDenseW.open("trained_dense_w.txt");
	if (!_trainDenseW.is_open()) 
	{
        cout << "Failed to open file trained_dense_w.\n";
    }
	for (int i = 0; i < 1568; i++) 
	{
		for (int j = 0; j < 120; j++) 
		{
			_trainDenseW << _denseW[i][j] << " ";
		}
		_trainDenseW << "\n\n";
	}
	_trainDenseW.close();

	_trainDenseB.open("trained_dense_b.txt");
	if (!_trainDenseB.is_open()) 
	{
        cout << "Failed to open file trained_dense_b.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		_trainDenseB << _denseB[i] << "\n";
	}
	_trainDenseB.close();

	_trainDenseW2.open("trained_dense_w2.txt");
	if (!_trainDenseW2.is_open()) 
	{
        cout << "Failed to open file trained_dense_w2.\n";
    }
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			_trainDenseW2 << _denseW2[i][j] << " ";
		}
		_trainDenseW2 << "\n\n";
	}
	_trainDenseW2.close();

	_trainDenseB2.open("trained_dense_b2.txt");
	if (!_trainDenseB2.is_open()) 
	{
        cout << "Failed to open file trained_dense_b2.\n";
    }
	for (int i = 0; i < 10; i++) 
	{
		_trainDenseB2 << _denseB2[i] << "\n";
	}
	_trainDenseB2.close();
}

void CNN::Forward(vector<vector<int>>& _img)
{
	/* Convolution Operation + Sigmoid Activation */
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				_maxPooling[filter_dim][i][j] = 0;
				_convLayer[filter_dim][i][j] = 0;
				_sigLayer[filter_dim][i][j] = 0;
				for (int k = 0; k < filter_size; k++) 
				{
					for (int l = 0; l < filter_size; l++) 
					{
						_convLayer[filter_dim][i][j] = _img[i + k][j + l] * _convW[filter_dim][k][l];
					}
				}
				_sigLayer[filter_dim][i][j] = Sigmoid(_convLayer[filter_dim][i][j] + _convB[filter_dim][i][j]);
			}
		}
	}
	
	/* MAX Pooling (max_pooling, max_layer) */
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
				cur_max = _sigLayer[filter_dim][i][j];
				for (int k = 0; k < 2; k++) 
				{
					for (int l = 0; l < 2; l++) 
					{
						if (_sigLayer[filter_dim][i + k][j + l] > cur_max) {
							max_i = i + k;
							max_j = j + l;
							cur_max = _sigLayer[filter_dim][max_i][max_j];
						}
					}
				}
				_maxPooling[filter_dim][max_i][max_j] = 1;
				_maxLayer[filter_dim][i / 2][j / 2] = cur_max;
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
				_denseInput[k] = _maxLayer[filter_dim][i][j];
				k++;
			}
		}
	}
	/* Dense Layer */
	for (int i = 0; i < 120; i++) 
	{
		_denseSum[i] = 0;
		_denseSigmoid[i] = 0;
		for (int j = 0; j < 1568; j++) 
		{
			_denseSum[i] += _denseW[j][i] * _denseInput[j];
		}
		_denseSum[i] += _denseB[i];
		_denseSigmoid[i] = Sigmoid(_denseSum[i]);
	}

	/* Dense Layer 2 */ 
	for (int i = 0; i < 10; i++) 
	{
		_denseSum2[i] = 0;
		for (int j = 0; j < 120; j++) 
		{
			_denseSum2[i] += _denseW2[j][i] * _denseSigmoid[j];
		}
		_denseSum2[i] += _denseB2[i];
	}

	/* Softmax Output */
	double den = SoftmaxDen(_denseSum2, 10);
	for (int i = 0; i < 10; i++) 
	{
		_denseSoftmax[i] = exp(_denseSum2[i]) / den;
	}
}

void CNN::Predict()
{
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
		GiveImg(_dataTest[i], img);
		Forward(img);
		int pre = GivePrediction();
		confusion_mat[_labelTest[i]][pre]++;
		if (pre == _labelTest[i])
		{
			cor++;
		}
	}
	float accu = double(cor) / val_len;
	cout << "Accuracy: " << accu << endl;

	cout << "    0  1  2  3  4  5  6  7  8  9" << endl;
	for (int i = 0; i < 10; i++) 
	{
		cout << i << ": ";
		for (int j = 0; j < 10; j++) 
		{
			// cout << confusion_mat[i][j] << " ";
			printf("%2d ", confusion_mat[i][j]);
		}
		cout << endl;
	}
}

void CNN::AdamOptimizer(double _lr, double _b1, double _b2, double _eps)
{
	vector<vector<vector<double>>> m3d;
	vector<vector<vector<double>>> v3d;
	vector<vector<double>> m2d;
	vector<vector<double>> v2d;
	vector<double> m1d(120, 0);
	vector<double> v1d(120, 0);
	double m_hat = 0;
	double v_hat = 0;
	for (int i = 0; i < 120; i++) 
	{
		m1d[i] = _b1 * _encDffB1[i] + (1 - _b1) * _encDffB1[i];
		v1d[i] = _b2 * _encDffB1[i] + (1 - _b2) * _encDffB1[i] * _encDffB1[i];
		m_hat = m1d[i] / (1 - pow(_b1, _t));
		v_hat = v1d[i] / (1 - pow(_b2, _t));
		_encDenseB[i] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);
	}
	m2d = vector<vector<double>>(120, vector<double>(10, 0));
	v2d = vector<vector<double>>(120, vector<double>(10, 0));
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{
			m1d[j] = _b1 * _encDffB2[j] + (1 - _b1) * _encDffB2[j];
			v1d[j] = _b2 * _encDffB2[j] + (1 - _b2) * _encDffB2[j] * _encDffB2[j];
			m_hat = m1d[j] / (1 - pow(_b1, _t));
			v_hat = v1d[j] / (1 - pow(_b2, _t));
			_encDenseB2[j] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);

			m2d[i][j] = _b1 * _encDffW2[i][j] + (1 - _b1) * _encDffW2[i][j];
			v2d[i][j] = _b2 * _encDffW2[i][j] + (1 - _b2) * _encDffW2[i][j] * _encDffW2[i][j];
			m_hat = m2d[i][j] / (1 - pow(_b1, _t));
			v_hat = v2d[i][j] / (1 - pow(_b2, _t));
			_encDenseW2[i][j] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);
		}
	}
	m2d = vector<vector<double>>(1568, vector<double>(120, 0));
	v2d = vector<vector<double>>(1568, vector<double>(120, 0));
	for (int i = 0; i < 120; i++) 
	{
		for (int k = 0; k < 1568; k++) 
		{
			m2d[k][i] = _b1 * _encDffW1[k][i] + (1 - _b1) * _encDffW1[k][i];
			v2d[k][i] = _b2 * _encDffW1[k][i] + (1 - _b2) * _encDffW1[k][i] * _encDffW1[k][i];
			m_hat = m2d[k][i] / (1 - pow(_b1, _t));
			v_hat = v2d[k][i] / (1 - pow(_b2, _t));
			_encDenseW[k][i] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);
		}
	}
	
	m3d = vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 0)));
	v3d = vector<vector<vector<double>>>(8, vector<vector<double>>(5, vector<double>(5, 0)));
	for (int i = 0; i < 8; i++) 
	{
		for (int k = 0; k < 5; k++) 
		{
			for (int j = 0; j < 5; j++) 
			{
				m3d[i][k][j] = _b1 * _encDffConvW[i][k][j] + (1 - _b1) * _encDffConvW[i][k][j];
				v3d[i][k][j] = _b2 * _encDffConvW[i][k][j] + (1 - _b2) * _encDffConvW[i][k][j] * _encDffConvW[i][k][j];
				m_hat = m3d[i][k][j] / (1 - pow(_b1, _t));
				v_hat = v3d[i][k][j] / (1 - pow(_b2, _t));
				_encConvW[i][k][j] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);
			}
		}
	}
	m3d = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	v3d = vector<vector<vector<double>>>(8, vector<vector<double>>(28, vector<double>(28, 0)));
	for (int i = 0; i < 8; i++) 
	{
		for (int l = 0; l < 28; l++) 
		{
			for (int m = 0; m < 28; m++) 
			{
				m3d[i][l][m] = _b1 * _encDffConvB[i][l][m] + (1 - _b1) * _encDffConvB[i][l][m];
				v3d[i][l][m] = _b2 * _encDffConvB[i][l][m] + (1 - _b2) * _encDffConvB[i][l][m] * _encDffConvB[i][l][m];
				m_hat = m3d[i][l][m] / (1 - pow(_b1, _t));
				v_hat = v3d[i][l][m] / (1 - pow(_b2, _t));
				_encConvB[i][l][m] -= _lr * m_hat / (sqrt(abs(v_hat)) + _eps);
			}
		}
	}
	_t++;
}
int main(int argc, char *argv[])
{
	int epoch = stoi(argv[1]);
	srand(time(NULL));
    CNN cnn;
	cnn.Init();
	cnn.Train(epoch);
	cnn.Predict();
    return 0;
}