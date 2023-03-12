#pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "IPlugPaths.h"
#include "CustomControls.h"
#include "NetworkControl.h"
#include "dsp.h"
#include "WeightSender.h"

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
  kCtrlNN,
  kCtrlTags
};

using namespace iplug;
using namespace igraphics;

class NNComp final : public Plugin
{
public:
  NNComp(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
public:
  void OnIdle() override;
  void OnReset() override;
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  
private:
  MeterSender<2> inSender {5., 0., 0.3, 0.5};
  MeterSender<2> outSender {5., 0., 0.3, 0.5};
  MeterSender<2> grSender {5., 0.1, 0.5, 0.5};
  WeightSender<32, 4> nnSender;
  
  NN<sample> nnL;
  NN<sample> nnR;
#endif
};
