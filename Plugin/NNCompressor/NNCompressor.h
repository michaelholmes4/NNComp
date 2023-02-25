#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "IPlugPaths.h"
#include "dsp.h"

const int kNumPresets = 1;

enum EParams
{
  kGain = 0,
  kOutGain,
  kModel,
  kBypass,
  kNumParams
};

enum EControlTags
{
  kCtrlInMeter = 0,
  kCtrlOutMeter,
  kCtrlGrMeter,
  kCtrlTags
};

using namespace iplug;
using namespace igraphics;

class NNCompressor final : public Plugin
{
public:
  NNCompressor(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
public:
  void OnIdle() override;
  void OnReset() override;
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  
private:
  IPeakAvgSender<2> inSender;
  IPeakAvgSender<2> outSender;
  ISender<1> grSender;
  
  NN<sample> nnL;
  NN<sample> nnR;
  float grBuffer;
  float grPrevious;
  float grAmount;
#endif
};
