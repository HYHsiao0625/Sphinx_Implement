#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <fstream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <ranges>
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

    /* MNIST Data */
    vector<vector<int> > _dataTrain;
    vector<vector<int> > _dataTest;
    vector<int> _labelTrain;
    vector<int> _labelTest;

    /* Weight */
    vector<vector<vector<double> > > _convW;
    vector<vector<vector<double> > > _convB;
	vector<vector<vector<double> > > _convLayer;
	vector<vector<vector<double> > > _sigLayer;
	vector<vector<vector<double> > > _maxPooling;
	vector<vector<vector<double> > > _maxLayer;
    vector<double> _denseInput;
	vector<vector<double> > _denseW;
	vector<double> _denseB;
	vector<double> _denseSum;
	vector<double> _denseSigmoid;
	vector<vector<double> > _denseW2;
	vector<double> _denseB2;
	vector<double> _denseSum2;
	vector<double> _denseSoftmax;

	/* Encrypt Params */
	CKKS _ckks;
	PublicKey _publickey;
	SecretKey _secretkey;
	Ciphertext _zero, _one, _eta, _cipherTemp;

    /* Encrypt Data */
	vector<vector<Ciphertext> > _encImg;
	vector<Ciphertext> _encLabel;

	/* Encrypt Weight */
    vector<vector<vector<Ciphertext> > > _encConvW;
	vector<vector<vector<Ciphertext> > > _encConvB;
	vector<vector<vector<Ciphertext> > > _encConvLayer;
	vector<vector<vector<Ciphertext> > > _encSigLayer;
	vector<vector<vector<Ciphertext> > > _encMaxPooling;
	vector<vector<vector<Ciphertext> > > _encMaxLayer;

	vector<Ciphertext> _encDenseInput;
	vector<vector<Ciphertext> > _encDenseW;
	vector<Ciphertext> _encDenseB;
	vector<Ciphertext> _encDenseSum;
	vector<Ciphertext> _encDenseSigmoid;
	vector<vector<Ciphertext> > _encDenseW2;
	vector<Ciphertext> _encDenseB2;
	vector<Ciphertext> _encDenseSum2;
	vector<Ciphertext> _encDenseSoftmax;

	vector<vector<Ciphertext> > _encDffW2;
	vector<Ciphertext> _encDffB2;
	vector<vector<Ciphertext> > _encDffW1;
	vector<Ciphertext> _encDffB1;
	vector<vector<vector<Ciphertext> > > _encDffMaxW;
	vector<vector<vector<Ciphertext> > > _encDffConvW;
	vector<vector<vector<Ciphertext> > > _encDffConvB;

public:
    CNN(/* args */);
    ~CNN();
	void Init();
	double Sigmoid(double x);
	double DffSigmoid(double x);
	double SoftmaxDen(vector<double> x, int len);
    void Train(int epochs);
    void ReadData();
    void PrintImg(vector<vector<double> > img);
    void InitializeWeight();
	void WriteInitWeight();
	
	void EncryptWeight();
	void DecryptWeight();
	void GiveImg(vector<int> vec, vector<vector<int> >& img);
	void GiveLabel(int y, vector<int>& vector_y);
	int GivePrediction();
	void EncryptData(vector<vector<int> >& img, vector<int>& label);
	void EncryptForward(vector<vector<Ciphertext> >& _encImg);
	void EncryptBackword(vector<Ciphertext>& y_hat, vector<Ciphertext>& y, vector<vector<Ciphertext> >& _encImg);
	void UpdateWeight();
	void WriteTrainedWeight();
	void Predict();
	void Forward(vector<vector<int> >& img);
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

	/* Train & Test Data */
	_dataTrain = vector<vector<int> >(60000, vector<int>(784, 0));
    _dataTest = vector<vector<int> > (10000, vector<int>(784, 0));
    _labelTrain = vector<int>(60000, 0);
    _labelTest = vector<int>(10000, 0);

    /* Weight */
    _convW = vector<vector<vector<double> > >(8, vector<vector<double> >(5, vector<double>(5, 0)));
    _convB = vector<vector<vector<double> > >(8, vector<vector<double> >(28, vector<double>(28, 0)));
    _convLayer = vector<vector<vector<double> > >(8, vector<vector<double> >(28, vector<double>(28, 0)));
	_sigLayer = vector<vector<vector<double> > >(8, vector<vector<double> >(28, vector<double>(28, 0)));
	_maxPooling = vector<vector<vector<double> > >(8, vector<vector<double> >(28, vector<double>(28, 0)));
	_maxLayer = vector<vector<vector<double> > >(8, vector<vector<double> >(28, vector<double>(28, 0)));
	
	_denseInput = vector<double>(1568, 0);
	_denseW = vector<vector<double> >(1568, vector<double>(120, 0));
	_denseB = vector<double>(120, 0);
	_denseSum = vector<double>(120, 0);
	_denseSigmoid = vector<double>(120, 0);
	_denseW2 = vector<vector<double> >(120, vector<double>(10, 0));
	_denseB2 = vector<double>(10, 0);
	_denseSum2 = vector<double>(10, 0);
	_denseSoftmax = vector<double>(10, 0);

	/* Encrypt Data */
	_encImg = vector<vector<Ciphertext> >(32, vector<Ciphertext>(32));
	_encLabel = vector<Ciphertext>(10);

	/* Encrypt Weight */
	_encConvW = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(5, vector<Ciphertext>(5)));
	_encConvB = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));
	_encConvLayer = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));
	_encSigLayer = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));
	_encMaxPooling = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));
	_encMaxLayer = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(14, vector<Ciphertext>(14)));

	_encDenseInput = vector<Ciphertext>(1568);
	_encDenseW = vector<vector<Ciphertext> >(1568, vector<Ciphertext>(120));
	_encDenseB = vector<Ciphertext>(120);
	_encDenseSum = vector<Ciphertext>(120);
	_encDenseSigmoid = vector<Ciphertext>(120);
	_encDenseW2 = vector<vector<Ciphertext> >(120, vector<Ciphertext>(10));
	_encDenseB2 = vector<Ciphertext>(10);
	_encDenseSum2 = vector<Ciphertext>(10);
	_encDenseSoftmax = vector<Ciphertext>(10);

	_encDffW2 = vector<vector<Ciphertext> >(120, vector<Ciphertext>(10));
	_encDffB2 = vector<Ciphertext>(10);
	_encDffW1 = vector<vector<Ciphertext> >(1568, vector<Ciphertext>(120));
	_encDffB1 = vector<Ciphertext>(120);
	_encDffMaxW = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));
	_encDffConvW = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(5, vector<Ciphertext>(5)));
	_encDffConvB = vector<vector<vector<Ciphertext> > >(8, vector<vector<Ciphertext> >(28, vector<Ciphertext>(28)));

	/* Generate Keyset */
	_ckks.initParams();
	_ckks.generateKey(&_publickey, &_secretkey);

	/* Generate Based Number with Cipher*/
	_ckks.encryptPlain(0, _publickey, &_zero);
	_ckks.encryptPlain(1, _publickey, &_one);
	_ckks.encryptPlain(eta, _publickey, &_eta);

	InitializeWeight();
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

void CNN::PrintImg(vector<vector<double> > _img)
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
void CNN::InitializeWeight()
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
	// convW
	for (size_t x = 0; x < _convW.size(); x++)
		for (size_t y = 0; y < _convW[x].size(); y++)
			for (size_t z = 0; z < _convW[x][y].size(); z++)
				_ckks.encryptPlain(_convW[x][y][z], _publickey, &_encConvW[x][y][z]);
	// convB
	for (size_t x = 0; x < _convB.size(); x++)
		for (size_t y = 0; y < _convB[x].size(); y++)
			for (size_t z = 0; z < _convB[x][y].size(); z++)
				_ckks.encryptPlain(_convB[x][y][z], _publickey, &_encConvB[x][y][z]);
	// denseW
	for (size_t x = 0; x < _denseW.size(); x++)
		for (size_t y = 0; y < _denseW[x].size(); y++)
				_ckks.encryptPlain(_denseW[x][y], _publickey, &_encDenseW[x][y]);
	// denseB
	for (size_t x = 0; x < _denseB.size(); x++)
		_ckks.encryptPlain(_denseB[x], _publickey, &_encDenseB[x]);
	// denseW2
	for (size_t x = 0; x < _denseW2.size(); x++)
		for (size_t y = 0; y < _denseW2[x].size(); y++)
			_ckks.encryptPlain(_denseW2[x][y], _publickey, &_encDenseW2[x][y]);
	// denseB2
    for (size_t x = 0; x < _denseB2.size(); x++)
		_ckks.encryptPlain(_denseB2[x], _publickey, &_encDenseB2[x]);
}

void CNN::DecryptWeight()
{
	/* Decrypt Weight */
	// convW
	for (size_t x = 0; x < _convW.size(); x++)
		for (size_t y = 0; y < _convW[x].size(); y++)
			for (size_t z = 0; z < _convW[x][y].size(); z++)
				_ckks.decryptCipher(_encConvW[x][y][z], _secretkey, &_convW[x][y][z]);
	// convB
	for (size_t x = 0; x < _convB.size(); x++)
		for (size_t y = 0; y < _convB[x].size(); y++)
			for (size_t z = 0; z < _convB[x][y].size(); z++)
				_ckks.decryptCipher(_encConvB[x][y][z], _secretkey, &_convB[x][y][z]);
	// denseW
	for (size_t x = 0; x < _denseW.size(); x++)
		for (size_t y = 0; y < _denseW[x].size(); y++)
				_ckks.decryptCipher(_encDenseW[x][y], _secretkey, &_denseW[x][y]);
	// denseB
	for (size_t x = 0; x < _denseB.size(); x++)
		_ckks.decryptCipher(_encDenseB[x], _secretkey, &_denseB[x]);
	// denseW2
	for (size_t x = 0; x < _denseW2.size(); x++)
		for (size_t y = 0; y < _denseW2[x].size(); y++)
			_ckks.decryptCipher(_encDenseW2[x][y], _secretkey, &_denseW2[x][y]);
	// denseB2
    for (size_t x = 0; x < _denseB2.size(); x++)
		_ckks.decryptCipher(_encDenseB2[x], _secretkey, &_denseB2[x]);
}

void CNN::GiveImg(vector<int> vec, vector<vector<int> >& img) 
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

void CNN::EncryptData(vector<vector<int> >& img, vector<int>& label)
{
	/* Encrypt Data */
	for (size_t x = 0; x < img.size(); x++)
		for (size_t y = 0; y < img[x].size(); y++)
			_ckks.encryptPlain(img[x][y], _publickey, &_encImg[x][y]);

	for (size_t i = 0; i < label.size(); i++)
		_ckks.encryptPlain(label[i], _publickey, &_encLabel[i]);
}

void CNN::EncryptForward(vector<vector<Ciphertext> >& _encImg)
{
	/* Convolution Operation + Sigmoid Activation */
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i++) 
		{
			for (int j = 0; j < 28; j++) 
			{
				_encConvLayer[filter_dim][i][j] = _zero;
				_encSigLayer[filter_dim][i][j] = _zero;
				_encMaxPooling[filter_dim][i][j] = _zero;
				for (int k = 0; k < filter_size; k++) 
				{
					for (int l = 0; l < filter_size; l++)
					{
						_cipherTemp = _encConvW[filter_dim][k][l];
						_ckks.evaluateCipher(&_encImg[i + k][j + l], "*", &_cipherTemp);
						_encConvLayer[filter_dim][i][j] = _cipherTemp;
						//_encConvLayer[filter_dim][i][j] = _encImg[i + k][j + l] * _encConvW[filter_dim][k][l];
					}
				}

				// Need decrypt before sigmoid, but here's for encrypt, move later
				vector<double> decrypted(3, 0);
				_ckks.decryptCipher(_encConvLayer[filter_dim][i][j], _secretkey, &decrypted[0]);
				_ckks.decryptCipher(_encConvB[filter_dim][i][j], _secretkey, &decrypted[1]);
				decrypted[2] = Sigmoid(decrypted[0] + decrypted[1]); 
				_ckks.encryptPlain(decrypted[2], _publickey, &_encSigLayer[filter_dim][i][j]);
			}
		}
		//PrintImg(_encSigLayer[filter_dim]);
	}
	
	// Need decrypt before pooling, but here's for encrypt, move later
	/* MAX Pooling (max_pooling, max_layer) */ 
	double cur_max = 0, tmp;
	int max_i = 0, max_j = 0;
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
	{
		for (int i = 0; i < 28; i += 2) 
		{
			for (int j = 0; j < 28; j += 2) 
			{
				max_i = i;
				max_j = j;
				_ckks.decryptCipher(_encSigLayer[filter_dim][i][j], _secretkey, &cur_max);
				// cur_max = _encSigLayer[filter_dim][i][j];
				for (int k = 0; k < 2; k++) 
				{
					for (int l = 0; l < 2; l++) 
					{	
						_ckks.decryptCipher(_encSigLayer[filter_dim][i + k][j + l], _secretkey, &tmp);
						if (tmp > cur_max)
						{
							max_i = i + k;
							max_j = j + l;
							_ckks.decryptCipher(_encSigLayer[filter_dim][max_i][max_j], _secretkey, &cur_max);
							// cur_max = _encSigLayer[filter_dim][max_i][max_j];
						}
					}
				}
				_encMaxPooling[filter_dim][max_i][max_j] = _one;
				_ckks.encryptPlain(cur_max, _publickey, &_encMaxLayer[filter_dim][i / 2][j / 2]);
				// _encMaxLayer[filter_dim][i / 2][j / 2] = cur_max;
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
		_encDenseSum[i] = _zero;
		_encDenseSigmoid[i] = _zero;
		for (int j = 0; j < 1568; j++) 
		{	
			_cipherTemp = _encDenseInput[j];
			_ckks.evaluateCipher(&_encDenseW[j][i], "*", &_cipherTemp);
			_ckks.evaluateCipher(&_cipherTemp, "+", &_encDenseSum[i]);
			// _encDenseSum[i] += _encDenseW[j][i] * _encDenseInput[j];
		}
		_ckks.evaluateCipher(&_encDenseB[i], "+", &_encDenseSum[i]);
		// _encDenseSum[i] += _encDenseB[i];
		double res;
		_ckks.decryptCipher(_encDenseSum[i], _secretkey, &res);
		res = Sigmoid(res);
		_ckks.encryptPlain(res, _publickey, &_encDenseSigmoid[i]);
		// _encDenseSigmoid[i] = Sigmoid(_encDenseSum[i]);
	}

	/* Dense Layer2 Computing */
	for (int i = 0; i < 10; i++) 
	{
		_encDenseSum2[i] = _zero;
		for (int j = 0; j < 120; j++)
		{	
			_cipherTemp = _encDenseSigmoid[j];
			_ckks.evaluateCipher(&_encDenseW2[j][i], "*", &_cipherTemp);
			_ckks.evaluateCipher(&_cipherTemp, "+", &_encDenseSum2[i]);
			// _encDenseSum2[i] += _encDenseW2[j][i] * _encDenseSigmoid[j];
		}
		_ckks.evaluateCipher(&_encDenseB2[i], "+", &_encDenseSum2[i]);
		// _encDenseSum2[i] += _encDenseB2[i];
	}

	/* Softmax Output */

	vector<double> decrypted;

	for (auto& i: _encDenseSum2) {
		double tmp;
		_ckks.decryptCipher(i, _secretkey, &tmp);
		decrypted.push_back(tmp);
	}

	double den = SoftmaxDen(decrypted, 10);

	for (int i = 0; i < decrypted.size(); i++) 
	{
		_ckks.encryptPlain(exp(decrypted[i]) / den, _publickey, &_encDenseSoftmax[i]);
	}
}

void CNN::EncryptBackword(vector<Ciphertext>& y_hat, vector<Ciphertext>& y, vector<vector<Ciphertext> >& _encImg) 
{
	vector<Ciphertext> _encDelta4;
	for (int i = 0; i < 10; i++) 
	{	
		_cipherTemp = y[i];
		_ckks.evaluateCipher(&y_hat[i], "-", &_cipherTemp);
		_encDelta4[i] = _cipherTemp;
		// _encDelta4[i] = y_hat[i] - y[i]; // Derivative of Softmax + Cross entropy
		_encDffB2[i] = _encDelta4[i]; // Bias Changes
		double _abs;
		_ckks.decryptCipher(_cipherTemp, _secretkey, &_abs);
		loss += abs(_abs);
	}

	// Calculate Weight Changes for Dense Layer 2
	for (int i = 0; i < 120; i++) 
	{
		for (int j = 0; j < 10; j++) 
		{	
			_cipherTemp = _encDelta4[j];
			_ckks.evaluateCipher(&_encDenseSigmoid[i], "*", &_cipherTemp);
			_encDffW2[i][j] = _cipherTemp;
		}
	}
	// Delta 3
	vector<Ciphertext> _encDelta3;
	for (int i = 0; i < 120; i++) 
	{
		_encDelta3[i] = _zero;
		for (int j = 0; j < 10; j++) 
		{	
			_cipherTemp = _encDelta4[j];
			_ckks.evaluateCipher(&_encDenseW2[i][j], "*", &_cipherTemp);
			_ckks.evaluateCipher(&_cipherTemp, "+",&_encDelta3[i]);
			// _encDelta3[i] += _encDenseW2[i][j] * _encDelta4[j];
		}
		double decrypted;
		_ckks.decryptCipher(_encDenseSum[i], _secretkey, &decrypted);
		decrypted = DffSigmoid(decrypted);
		_ckks.encryptPlain(decrypted, _publickey, &_cipherTemp);
		_ckks.evaluateCipher(&_cipherTemp, "*", &_encDelta3[i]);
		// _encDelta3[i] *= DffSigmoid(decrypted);
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
			_cipherTemp = _encDelta3[j];
			_ckks.evaluateCipher(&_encDenseInput[i], "*", &_cipherTemp);
			_encDffW1[i][j] = _cipherTemp;
			// _encDffW1[i][j] = _encDenseInput[i] * _encDelta3[j];
		}
	}

	// _encDelta2
	vector<Ciphertext> _encDelta2;
	for (int i = 0; i < 1568; i++) 
	{
		_encDelta2[i] = _zero;
		for (int j = 0; j < 120; j++) 
		{	
			_cipherTemp = _encDelta3[j];
			_ckks.evaluateCipher(&_encDenseW[i][j], "*", &_cipherTemp);
			_ckks.evaluateCipher(&_cipherTemp, "+", &_encDelta2[i]);
			// _encDelta2[i] += _encDenseW[i][j] * _encDelta3[j];
		}
		
		double decrypted;
		_ckks.decryptCipher(_encDenseInput[i], _secretkey, &decrypted);
		decrypted = DffSigmoid(decrypted);
		_ckks.encryptPlain(decrypted, _publickey, &_cipherTemp);
		_ckks.evaluateCipher(&_cipherTemp, "*", &_encDelta2[i]);
		// _encDelta2[i] *= DffSigmoid(_encDenseInput[i]);
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
						double decrypted;
						_ckks.decryptCipher(_encMaxPooling[filter_dim][i + l][j + m], _secretkey, &decrypted);
						if (decrypted == 1) 
						{
							_encDffMaxW[filter_dim][i][j] = _encDelta2[k];
						}
						else
						{
							_encDffMaxW[filter_dim][i][j] = _zero;
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
				_encDffConvW[filter_dim][i][j] = _zero;
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
				Ciphertext cur_val = _encDffMaxW[filter_dim][i][j];
				for (int k = 0; k < 5; k++) 
				{
					for (int l = 0; l < 5; l++) 
					{	
						_ckks.evaluateCipher(&_encImg[i + k][j + l], "*", &cur_val);
						_ckks.evaluateCipher(&cur_val, "+", &_encDffConvW[filter_dim][k][l]);
						// _encDffConvW[filter_dim][k][l] += _encImg[i + k][j + l] * cur_val;
					}
				}
			}
		}
	}
}

void CNN::UpdateWeight()
{
	for (int i = 0; i < 120; i++) 
	{	
		_cipherTemp = _encDffB1[i];
		_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
		_ckks.evaluateCipher(&_encDenseB[i], "-", &_cipherTemp);
		_encDenseB[i] = _cipherTemp;
		// _encDenseB[i] -= eta * _encDffB1[i];

		for (int j = 0; j < 10; j++) 
		{	
			_cipherTemp = _encDffB2[j];
			_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
			_ckks.evaluateCipher(&_encDenseB2[j], "-", &_cipherTemp);
			_encDenseB2[j] = _cipherTemp;
			//_encDenseB2[j] -= eta * _encDffB2[j];
			_cipherTemp = _encDffW2[i][j];
			_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
			_ckks.evaluateCipher(&_encDenseW2[i][j], "-", &_cipherTemp);
			_encDenseW2[i][j] = _cipherTemp;
			// _encDenseW2[i][j] -= eta * _encDffW2[i][j];
		}
		for (int k = 0; k < 1568; k++) 
		{	
			_cipherTemp = _encDffW1[k][i];
			_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
			_ckks.evaluateCipher(&_encDenseW[k][i], "-", &_cipherTemp);
			_encDenseW[k][i] = _cipherTemp;
			// a -= b --> a = a - b
			// _encDenseW[k][i] -= eta * _encDffW1[k][i];
		}
	}

	for (int i = 0; i < 8; i++) 
	{
		for (int k = 0; k < 5; k++) 
		{
			for (int j = 0; j < 5; j++) 
			{	
				_cipherTemp = _encDffConvW[i][k][j];
				_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
				_ckks.evaluateCipher(&_encConvW[i][k][j], "-", &_cipherTemp);
				_encConvW[i][k][j] = _cipherTemp;
				// _encConvW[i][k][j] -= eta * _encDffConvW[i][k][j];
			}
		}
		for (int l = 0; l < 28; l++) 
		{
			for (int m = 0; m < 28; m++) 
			{	
				_cipherTemp = _encDffConvB[i][l][m];
				_ckks.evaluateCipher(&_eta, "*", &_cipherTemp);
				_ckks.evaluateCipher(&_encConvB[i][l][m], "-", &_cipherTemp);
				_encConvB[i][l][m] = _cipherTemp;
				// _encConvB[i][l][m] -= eta * _encDffConvB[i][l][m];
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
		for (int j = 0; j < batch_size; j++)
		{
			cout << "\rEpoch: " << i << " --------- |" << setw(3) << j << " / " << batch_size << " | Loss: " << fixed << Loss << " |" << flush;
			num = rand() % 60000;
			vector<vector<int> > img(32, vector<int>(32, 0));
			vector<int> label(10, 0);
			GiveLabel(_labelTrain[num], label);
			GiveImg(_dataTrain[num], img);
			EncryptData(img, label);
			EncryptForward(_encImg);
			EncryptBackword(_encDenseSoftmax, _encLabel, _encImg);
			UpdateWeight();
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

void CNN::Forward(vector<vector<int> >& _img)
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
		vector<vector<int> > img(32, vector<int>(32, 0));
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
}

int main()
{
	srand(time(NULL));
    CNN cnn;
	cnn.Init();
	cnn.Train(100);
	cnn.Predict();
    return 0;
}