#include "NNComp.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"
#include "IPlugPaths.h"
#include "IVMeterControl.h"


NNComp::NNComp(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kGain)->InitDouble("InGain",0, -40., 40., 0.1, "");
  GetParam(kOutGain)->InitDouble("OutGain",0.0, -40., 40., 0.1, "");
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
    //Resources and misc
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_WHITE);
    pGraphics->LoadFont("Inter-Regular", INTER_FN);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    
    //Bounds
    const IRECT b = pGraphics->GetBounds();
    const IRECT bLeftPanel = b.GetInsideSize(7, 224, 70, 294);
    const IRECT bRightPanel = b.GetInsideSize(759, 224, 70, 294);
    const IRECT bMainPanel = b.GetInsideSize(87, 97, 662, 559);
    const IRECT bInputMeterArea = bLeftPanel.GetCentredInside(IRECT(0,0,52,275));
    const IRECT bInputMeter = bInputMeterArea.GetFromLeft(52).GetFromTop(186);
    const IRECT bOutputMeterArea = bRightPanel.GetCentredInside(IRECT(0,0,52,275));
    const IRECT bOutputMeter = bOutputMeterArea.GetFromRight(52).GetFromTop(186);
    const IRECT bInputKnob = bInputMeterArea.GetReducedFromTop(201).GetReducedFromLeft(8).GetReducedFromRight(8);
    const IRECT bOutputKnob = bOutputMeterArea.GetReducedFromTop(201).GetReducedFromLeft(8).GetReducedFromRight(8);
    
    //Controls
    pGraphics->AttachControl(new BackgroundControl(b));
    pGraphics->AttachControl(new MeterControl(bInputMeter), kCtrlInMeter);
    pGraphics->AttachControl(new MeterControl(bOutputMeter), kCtrlOutMeter);
    pGraphics->AttachControl(new KnobControl(bInputKnob, kGain, "Input"));
    pGraphics->AttachControl(new KnobControl(bOutputKnob, kOutGain, "Output"));
    
  };
#endif
}

#if IPLUG_DSP

void NNComp::OnIdle()
{
  inSender.TransmitData(*this);
  outSender.TransmitData(*this);
  //grSender.TransmitData(*this);
}

void NNComp::OnReset()
{
  inSender.Reset(GetSampleRate());
  outSender.Reset(GetSampleRate());
}

void NNComp::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  const double inGain = DBToAmp(GetParam(kGain)->Value());
  const double outGain = DBToAmp(GetParam(kOutGain)->Value());
  const int nChans = NOutChansConnected();
  const int model = GetParam(kModel)->Int();


  //Iterate over buffer
  for (int s = 0; s < nFrames; s++) {
    for (int c = 0; c < nChans; c++) {
      grBuffer = 0.0;
      //apply NN
      inputs[c][s] *= inGain;
      if(c == 0) {
        //nnL.ProcessSample(&inputs[c][s], &outputs[c][s], model);
      } else if(c == 1) {
        //nnR.ProcessSample(&inputs[c][s], &outputs[c][s], model);
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
  //grSender.PushData({kCtrlGrMeter, {grAmount}});
}
#endif
