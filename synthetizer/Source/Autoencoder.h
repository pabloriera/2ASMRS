/*
  ==============================================================================
    Autoencoder.h
    Created: 18 Aug 2020 11:29:03pm
    Author:  Joaquin Romera
    Author:  Pytorch version Pablo Riera
  ==============================================================================
*/

#pragma once

#include <complex>
#include <fstream>
#include <torch/script.h> // One-stop header.
#include <JuceHeader.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>
#define FFT_SIZE 8192

struct SliderMinMax
{
    float min;
    float max;
};

struct Parameters
{
    Parameters(float pXMax, float pSClip, const unsigned int pWinLength,
               const unsigned int pLatentDim, const nlohmann::json &PZRange) : xMax(pXMax), sClip(pSClip), winLength(pWinLength), latentDim(pLatentDim)
    {
        for (const auto &r : PZRange)
            zRange.push_back({r["min"], r["max"]});


    }

    float xMax;
    float sClip;
    unsigned int winLength;
    unsigned int latentDim;
    std::vector<SliderMinMax> zRange;
};

struct Ztrack
{
    Ztrack(const auto &PZtrack)
    {
        for (const auto &r : PZtrack)
        {
            z.push_back(r);
        }
    }
    std::vector<float> z;
};

class Autoencoder
{

private:
    torch::jit::script::Module mAutoencoder;

    std::vector<float> mInput;
    std::vector<torch::jit::IValue> mInputTensor;

    Parameters *mParams;
    // Ztrack *ztrack;
      // Create a vector of vectors to store the data.

  
    float fftArray[FFT_SIZE];
    unsigned int rfftSize;
    unsigned int index;
    unsigned int idxProc;
    float phaseg=0;
    std::vector<float> mAudio;

    juce::Random random;
    juce::dsp::FFT mFFT = {12};

public:
    std::vector<std::vector<float>> ztrack;

    Autoencoder(const std::string &path)
    {
        std::ifstream ifs(path);
        nlohmann::json settings;
        ifs >> settings;

        mParams = new Parameters(
            settings["parameters"]["xMax"],
            settings["parameters"]["sClip"],
            settings["parameters"]["win_length"],
            settings["parameters"]["latent_dim"],
            settings["parameters"]["zRange"]);

        // ztrack = new Ztrack(settings["ztrack"]);
        // print content of ztrack
        // for (const auto &r : ztrack->z)
        //     DBG(r);

        auto lists = settings["ztrack"];

        for (auto list : lists) {
            ztrack.push_back(list);
        }


        DBG(settings["model_path"].get<std::string>());

        try
        {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            
            mAutoencoder = torch::jit::load(settings["model_path"].get<std::string>());
            // Create a vector of inputs.
            mInputTensor.push_back(torch::ones({1, mParams->latentDim}));

            DBG("[AUTOENCODER] INPUT SIZES: " << mInputTensor[0].toTensor().size(1));

            // Execute the model and turn its output into a tensor.
            at::Tensor output = mAutoencoder.forward(mInputTensor).toTensor();
            DBG("[AUTOENCODER] OUTPUT SIZES: " << output.size(1));
            DBG("[AUTOENCODER] WIN SIZE: " <<  (int)mParams->winLength);

            rfftSize = output.sizes()[1];
            index = 0;
            idxProc = 0;
            mAudio.resize(mParams->winLength, 0);

            // // // TODO fix how to initialize values to avoid pops in audio when loading
            mParams->xMax = 0;
            mParams->sClip = -100;

            for (int i=0;i<FFT_SIZE;i++)
            {
                fftArray[i]=0.0;
            }
        }
        catch (const c10::Error &e)
        {
            std::cerr << "error loading the model\n";
        }
    }

    ~Autoencoder()
    {
        DBG("[AUTOENCODER] Destroying...");
    }

    void play(size_t ix, std::vector<juce::Slider *> &mSliders)
    {
        DBG("[AUTOENCODER] Play");
        // Iterate over ztrack vector and change the slider value with the elements
        // for (int i = 0; i < ztrack.size(); i++) 
        // {
            for (int dim = 0; dim < mParams->latentDim; dim++)
            {
                const float val = ztrack[ix][dim];
                mInputTensor[0].toTensor().index_put_({0, (long int) dim}, val);
                mSliders[dim]->setValue(val);
            }
        // }
    }

    void setInputLayers(const size_t pos, const float newValue)
    {
        DBG("[AUTOENCODER] slider: " << pos << " new value: " << newValue);
        mInputTensor[0].toTensor().index_put_({0, (long int) pos}, newValue);
    }

    size_t getInputDepth() const
    {
        return mInputTensor[0].toTensor().size(1);
    }

    float getInputTensorAt(int i)
    {
        return mInputTensor[0].toTensor().index({0, i}).item<float>();
    }

    std::pair<float, float> getSlider(const size_t pos)
    {
        return {mParams->zRange[pos].min, mParams->zRange[pos].max};
    }

    void setXMax(const float newValue)
    {
        DBG("[AUTOENCODER] xMax Length: " << newValue);
        mParams->xMax = newValue;
    }

    void setSClip(const float newValue)
    {
        DBG("[AUTOENCODER] sClip Length: " << newValue);
        mParams->sClip = newValue;
    }

    void getNextAudioBlock(const juce::AudioSourceChannelInfo &bufferToFill)
    {
        calculate(bufferToFill.numSamples);

        for (int sample = 0; sample < bufferToFill.numSamples; ++sample)
        {
            bufferToFill.buffer->setSample(0, sample, mAudio[index + sample]);
            bufferToFill.buffer->setSample(1, sample, mAudio[index + sample]);            
            mAudio[index + sample] = 0;
        }

        index += bufferToFill.numSamples;
        if (index>=mParams->winLength)
            index=0;
    }

    void calculate(const int bufferToFillSize)
    {

        at::Tensor predictionResult = mAutoencoder.forward(mInputTensor).toTensor();

        for (int i = 0; i < rfftSize; ++i)
        {
            const float tmp = predictionResult.index({0,i}).item<float>();
            const float power = (( tmp*mParams->xMax) + mParams->sClip) / 10;
            float y_aux = std::sqrt(std::pow(10, power));
            float phase = random.nextFloat() * juce::MathConstants<float>::twoPi;
            fftArray[2 * i] = y_aux * std::cos(phase);
            fftArray[(2 * i) + 1] = y_aux * std::sin(phase);
        }

        mFFT.performRealOnlyInverseTransform(fftArray);

        for (int i = 0; i < mParams->winLength; ++i)
        {
            // TODO: precalculate cosine window table 
            const float multiplier = 0.5f * (1 - std::cos(juce::MathConstants<float>::twoPi * i / (mParams->winLength - 1)));
            const float sample = fftArray[i] * multiplier;
            const int idx = (idxProc + i) % mParams->winLength;
            mAudio[idx] += sample;
        }

        idxProc += bufferToFillSize;
        if (idxProc>=mParams->winLength)
            idxProc=0;
    }
};