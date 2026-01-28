#pragma once
#include <sstream>

class BenchmarkData
{
private:
    char *timestamp;
    char *framework;
    char *workload_name;
    char *workload_type;
    char *device;
    int batch_size;
    int input_size;
    char *optimizer;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float time_ms;
    int batch_index;
    float loss;

public:
    BenchmarkData()
    {
    }
    BenchmarkData(char *ts, char *fw, char *w_name, char *w_type, char *dev,
                  int b_size, int in_size,
                  char *opt, float lr, float b1, float b2,
                  float eps, float t_ms, int b_index, float l)
    {
        timestamp = ts;
        framework = fw;
        workload_name = w_name;
        workload_type = w_type;
        device = dev;
        batch_size = b_size;
        input_size = in_size;
        optimizer = opt;
        learning_rate = lr;
        beta1 = b1;
        beta2 = b2;
        epsilon = eps;
        time_ms = t_ms;
        batch_index = b_index;
        loss = l;
    }
    ~BenchmarkData()
    {
    }

    void setTimestamp(char *ts) { timestamp = ts; }
    void setFramework(char *fw) { framework = fw; }
    void setWorkloadName(char *w_name) { workload_name = w_name; }
    void setWorkloadType(char *w_type) { workload_type = w_type; }
    void setDevice(char *dev) { device = dev; }
    void setBatchSize(int b_size) { batch_size = b_size; }
    void setInputSize(int in_size) { input_size = in_size; }
    void setOptimizer(char *opt) { optimizer = opt; }
    void setLearningRate(float lr) { learning_rate = lr; }
    void setBeta1(float b1) { beta1 = b1; }
    void setBeta2(float b2) { beta2 = b2; }
    void setEpsilon(float eps) { epsilon = eps; }
    void setTimeMs(float t_ms) { time_ms = t_ms; }
    void setBatchIndex(int b_index) { batch_index = b_index; }
    void setLoss(float l) { loss = l; }

    const char *getCSVHeader()
    {
        return "timestamp,framework,workload_name,workload_type,device,batch_size,input_size,optimizer,learning_rate,beta1,beta2,epsilon,time_ms";
    }

    const char *toCSVString()
    {
        std::stringstream ss;
        ss << timestamp << "," << framework << "," << workload_name << ","
           << workload_type << "," << device << "," << batch_size << ","
           << input_size << "," << "," << optimizer << ","
           << learning_rate << "," << beta1 << "," << beta2 << ","
           << epsilon << "," << time_ms << "," << batch_index << "," << loss;
        return ss.str().c_str();
    }
};
