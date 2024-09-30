#ifndef CKKS_HPP
#define CKKS_HPP

using namespace std;
using namespace seal;

class CKKS {
    private:
        EncryptionParameters _param;
        
    public:
        void initParams();
        void generateKey(PublicKey* publickey, SecretKey* secretkey);
        void encryptPlain(const double plaintext, const PublicKey& publickey, Ciphertext* ciphertext);
        void evaluateCipher(Ciphertext* arg1, const char* specfied_operator, Ciphertext* arg2 = nullptr);
        void decryptCipher(const Ciphertext& ciphertext, const SecretKey& secretkey, double* result);
};

#endif