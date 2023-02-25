#include "NNCompressor.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"
#include "IPlugPaths.h"
#include "IVMeterControl.h"

NNCompressor::NNCompressor(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kGain)->InitDouble("InGain",-30.0, -40., 40., 0.001, "dB");
  GetParam(kOutGain)->InitDouble("OutGain",30.0, -40., 40., 0.001, "dB");
  GetParam(kModel)->InitEnum("ModelSelect", 0, 26, "", IParam::kFlagsNone, "",
                             "gru-32-1",
                             "gru-16-2",
                             "gru-8-4",
                             "gru-16-1",
                             "gru-8-2",
                             "gru-4-4",
                             "gru-8-1",
                             "gru-4-2",
                             "gru-2-4",
                             "lstm-32-1",
                             "lstm-16-2",
                             "lstm-8-4",
                             "lstm-16-1",
                             "lstm-8-2",
                             "lstm-4-4",
                             "lstm-8-1",
                             "lstm-4-2",
                             "lstm-2-4",
                             "rnn-32-1",
                             "rnn-16-2",
                             "rnn-8-4",
                             "rnn-16-1",
                             "rnn-8-2",
                             "rnn-4-4",
                             "rnn-8-1",
                             "rnn-4-2",
                             "rnn-2-4"
                             );
  GetParam(kBypass)->InitBool("Bypass", true);

#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_WHITE);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    
    //Bounds
    const IRECT b = pGraphics->GetBounds();
    const IRECT bTitle = b.GetFromTop(20.);
    const IRECT bFooter = b.GetFromBottom(10);
    const IRECT bPlugin = b.GetReducedFromBottom(10).GetFromBottom(PLUG_HEIGHT - 10 - 20);
    const IRECT bInMeter = bPlugin.GetFromLeft(30).GetHShifted(10);
    const IRECT bOutMeter = bPlugin.GetFromRight(30).GetHShifted(-10);
    const IRECT bInKnob = bInMeter.GetCentredInside(80).GetHShifted(50);
    const IRECT bOutKnob = bOutMeter.GetCentredInside(80).GetHShifted(-50);
    const IRECT bModelSelect = bPlugin.GetCentredInside(100).GetReducedFromTop(30).GetReducedFromBottom(30);
    const IRECT bGrMeter = bPlugin.GetCentredInside(100).GetReducedFromTop(40).GetReducedFromBottom(40).GetVShifted(50);
    const IRECT bBypass = bModelSelect.GetVShifted(-60).GetHPadded(-30);
    
    //Elements
    pGraphics->AttachControl(new ITextControl(bTitle, "Real-Time VA Modelling of an Audio Compressor Using Deep Neural Networks", IText(20)));
    pGraphics->AttachControl(new IVPeakAvgMeterControl<2>(bInMeter, "InMeter", DEFAULT_STYLE.WithShowLabel(false).WithShowValue(true)), kCtrlInMeter);
    pGraphics->AttachControl(new IVPeakAvgMeterControl<2>(bOutMeter, "OutMeter", DEFAULT_STYLE.WithShowLabel(false).WithShowValue(true)), kCtrlOutMeter);
    pGraphics->AttachControl(new IVKnobControl(bInKnob, kGain, "Input"));
    pGraphics->AttachControl(new IVKnobControl(bOutKnob, kOutGain, "Output"));
    pGraphics->AttachControl(new ITextControl(bFooter, "Michael Holmes 2022", IText(10)));
    pGraphics->AttachControl(new ICaptionControl(bModelSelect, kModel));
    pGraphics->AttachControl(new IVLabelControl(bModelSelect.GetVShifted(-30).GetHPadded(40),"Neural Network Architecture", DEFAULT_STYLE.WithDrawFrame(false).WithDrawShadows(false)));
    pGraphics->AttachControl(new IVMeterControl<1>(bGrMeter, "Gain Reduction", DEFAULT_STYLE.WithShowValue(false).WithShowLabel(false), EDirection::Horizontal, {""}, 0, IVMeterControl<>::EResponse::Linear, -20., 0.), kCtrlGrMeter);
    pGraphics->AttachControl(new IVLabelControl(bGrMeter.GetVShifted(20),"Gain Reduction", DEFAULT_STYLE.WithDrawFrame(false).WithDrawShadows(false)));
    pGraphics->AttachControl(new IVToggleControl(bBypass, kBypass, "", DEFAULT_STYLE.WithShowLabel(false)));
  };
#endif
}

#if IPLUG_DSP

void NNCompressor::OnIdle()
{
  inSender.TransmitData(*this);
  outSender.TransmitData(*this);
  grSender.TransmitData(*this);
}

void NNCompressor::OnReset()
{
  inSender.Reset(GetSampleRate());
  outSender.Reset(GetSampleRate());
}

void NNCompressor::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const double inGain = DBToAmp(GetParam(kGain)->Value());
  const double outGain = DBToAmp(GetParam(kOutGain)->Value());
  const int nChans = NOutChansConnected();
  const int model = GetParam(kModel)->Int();
  const bool bypass = GetParam(kBypass)->Bool();

  //Iterate over buffer
  for (int s = 0; s < nFrames; s++) {
    for (int c = 0; c < nChans; c++) {
      grBuffer = 0.0;
      //apply NN
      inputs[c][s] *= inGain;
      
      if(bypass) {
        if(c == 0) {
          nnL.ProcessSample(&inputs[c][s], &outputs[c][s], model);
        } else if(c == 1) {
          nnR.ProcessSample(&inputs[c][s], &outputs[c][s], model);
        }
      }
      
      //GR Meter
      grBuffer += abs(inputs[c][s]) - abs(outputs[c][s]);
      
      //Apply out gain
      outputs[c][s] *= outGain;
    }
    
    grBuffer /= nChans;
    if(grBuffer > grPrevious) {
      grPrevious = grBuffer;
    } else {
      grPrevious -= 1 / (1 * GetSampleRate());
    }
  }
  
  //Send values to meters
  grAmount = 1 - grPrevious;
  inSender.ProcessBlock(inputs, nFrames, kCtrlInMeter);
  outSender.ProcessBlock(outputs, nFrames, kCtrlOutMeter);
  grSender.PushData({kCtrlGrMeter, {grAmount}});
}
#endif
