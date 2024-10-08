#ifndef CKKS_HPP
#define CKKS_HPP

using namespace std;
using namespace seal;

class CKKS {
    private:
        EncryptionParameters _param;
        RelinKeys _relinkeys;
        
    public:
        void initParams();
        void generateKey(PublicKey* publickey, SecretKey* secretkey);
        void Encoder(const double source, Plaintext* result);
        void Decoder(const Plaintext source, double* result);
        void encryptPlain(const double plaintext, const PublicKey& publickey, Ciphertext* ciphertext);
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr);
        void evaluatePlain(Ciphertext* arg1, const char* specfied_operator, const Plaintext* arg2);
        void decryptCipher(const Ciphertext& ciphertext, const SecretKey& secretkey, double* result);
};

#endif