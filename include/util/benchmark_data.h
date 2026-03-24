#pragma once
#include <sstream>
#include <string>

class BenchmarkData
{
public:
    const char *timestamp = "";
    const char *device_name = "";
    const char *framework = "";
    const char *workload_name = "";
    const char *workload_type = "";
    const char *device = "";
    int batch_size = 0;
    long input_size = 0;
    const char *optimizer = "";
    float learning_rate = 0.0f;
    float beta1 = 0.0f;
    float beta2 = 0.0f;
    float epsilon = 0.0f;
    float time_ms = 0.0f;
    int batch_index = 0;
    float loss = 0.0f;
    float accuracy = 0.0f;

    BenchmarkData()
    {
    }
    BenchmarkData(const char *ts, const char *device_name, const char *fw, const char *w_name, const char *w_type, const char *dev,
                  int b_size, long in_size,
                  const char *opt, float lr, float b1, float b2,
                  float eps, float t_ms, int b_index, float l, float acc = 0.0f)
    {
        this->timestamp = ts;
        this->device_name = device_name;
        this->framework = fw;
        this->workload_name = w_name;
        this->workload_type = w_type;
        this->device = dev;
        this->batch_size = b_size;
        this->input_size = in_size;
        this->optimizer = opt;
        this->learning_rate = lr;
        this->beta1 = b1;
        this->beta2 = b2;
        this->epsilon = eps;
        this->time_ms = t_ms;
        this->batch_index = b_index;
        this->loss = l;
        this->accuracy = acc;
    }
    ~BenchmarkData()
    {
    }

    static constexpr const char *csv_header()
    {
        return "timestamp,device_name,framework,workload_name,workload_type,device,batch_size,input_size,optimizer,learning_rate,beta1,beta2,epsilon,time_ms,batch_index,loss,accuracy";
    }

    std::string to_csv_string() const
    {
        std::stringstream ss;
        auto safe = [](const char *s)
        { return s ? s : ""; };
        ss << safe(timestamp) << "," << safe(device_name) << "," << safe(framework) << "," << safe(workload_name) << ","
           << safe(workload_type) << "," << safe(device) << "," << batch_size << ","
           << input_size << "," << safe(optimizer) << ","
           << learning_rate << "," << beta1 << "," << beta2 << ","
           << epsilon << "," << time_ms << "," << batch_index << "," << loss << "," << accuracy;
        return ss.str();
    }
};
