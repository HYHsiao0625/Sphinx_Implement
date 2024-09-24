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

    /* Encrypt Data */
	vector<vector<int>> _encImg;
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
	void EncryptForward(vector<vector<int>>& _encImg);
	void EncryptBackword(vector<double>& y_hat, vector<int>& y, vector<vector<int>>& _encImg);
	void UpdateWeight();
	void WriteTrainedWeight();
	void Predict();
	void Forward(vector<vector<int>>& img);
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

	/* Encrypt Data */
	_encImg = vector<vector<int>>(32, vector<int>(32, 0));
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
double CNN::Sigmoid(double _x) 
{
	if (_x > 500)
	{
		_x = 500;
	}
	if (_x < -500)
	{
		_x = -500;
	}
	return 1 / (1 + exp(-_x));
}

double CNN::DffSigmoid(double _x) 
{
	double _sig = Sigmoid(_x);
	return _sig * (1 - _sig);
}

double CNN::SoftmaxDen(vector<double> _x, int _len) 
{
	double _val = 0;
	for (int i = 0; i < _len; i++)
	{
		_val += exp(_x[i]);
	}
	return _val;
}

void CNN::ReadData()
{
    /* Read Train Data */
    ifstream _csvread;
    _csvread.open("../mnist_train.csv", ios::in);
    if (_csvread) 
    {
        //Datei bis Ende einlesen und bei ';' strings trennen
        string _s;
        int _dataPt = 0;
        while (getline(_csvread, _s)) 
        {
            stringstream _ss(_s);
            int _pxl = 0;
            while (_ss.good()) 
            {
                string _substr;
                getline(_ss, _substr, ',');
                if (_pxl == 0) 
                {
                    _labelTrain[_dataPt] = stoi(_substr);
                }
                else 
                {
                    _dataTrain[_dataPt][_pxl - 1] = stoi(_substr);
                }
                _pxl++;
            }
            _dataPt++;
        }
        _csvread.close();
    }
    else 
    {
        //cerr << "Fehler beim Lesen!" << endl;
        cerr << "Can not read data!" << endl;
    }
    
    /* Read Test Data */
    _csvread.open("../mnist_test.csv", ios::in);
    if (_csvread) 
    {
        //Datei bis Ende einlesen und bei ';' strings trennen
        string _s;
        int _dataPt = 0;
        while (getline(_csvread, _s)) 
        {
            stringstream _ss(_s);
            int _pxl = 0;
            while (_ss.good()) 
            {
                string _substr;
                getline(_ss, _substr, ',');
                if (_pxl == 0) 
                {
                    _labelTest[_dataPt] = stoi(_substr);
                }
                else 
                {
                    _dataTest[_dataPt][_pxl - 1] = stoi(_substr);
                }
                _pxl++;
            }
            _dataPt++;
        }
        _csvread.close();
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

void CNN::GiveImg(vector<int> _vec, vector<vector<int>>& _img) 
{
	int k = 0;
	for (int i = 0; i < 32; i++) 
	{
		for (int j = 0; j < 32; j++) 
		{
			if (i < 2 || j < 2 || i > 29 || j > 29) 
			{
				_img[i][j] = 0;
			}
			else 
			{
				_img[i][j] = _vec[k];
				k++;
			}
		}
	}
}

void CNN::GiveLabel(int _y, vector<int>& _label) 
{
	for (int i = 0; i < 10; i++)
	{
		_label[i] = 0;
	}
	_label[_y] = 1;
}

int CNN::GivePrediction()
{
	double _maxVal = _denseSoftmax[0];
	int _maxPos = 0;
	for (int i = 1; i < 10; i++) 
	{
		if (_denseSoftmax[i] > _maxVal)
		{
			_maxVal = _denseSoftmax[i];
			_maxPos = i;
		}
	}

	return _maxPos;
}

void CNN::EncryptData(vector<vector<int>>& _img, vector<int>& _label)
{
	/* Encrypt Data */
	_encImg = _img;
	_encLabel = _label;
}

void CNN::EncryptForward(vector<vector<int>>& _encImg)
{
	/* Convolution Operation + Sigmoid Activation */
	for (int filter_dim = 0; filter_dim < 8; filter_dim++) 
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
						_encConvLayer[filter_dim][i][j] = _encImg[i + k][j + l] * _encConvW[filter_dim][k][l];
					}
				}
				_encSigLayer[filter_dim][i][j] = Sigmoid(_encConvLayer[filter_dim][i][j] + _encConvB[filter_dim][i][j]);
			}
		}
		//PrintImg(_encSigLayer[filter_dim]);
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

void CNN::EncryptBackword(vector<double>& y_hat, vector<int>& y, vector<vector<int>>& _encImg) 
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
						_encDffConvW[filter_dim][k][l] += _encImg[i + k][j + l] * cur_val;
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
		for (int j = 0; j < batch_size; j++)
		{
			cout << "\rEpoch: " << i << " --------- |" << setw(3) << j << " / " << batch_size << " | Loss: " << fixed << Loss << " |" << flush;
			num = rand() % 60000;
			vector<vector<int>> img(32, vector<int>(32, 0));
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