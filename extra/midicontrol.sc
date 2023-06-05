
MIDIClient.init;
~midiOut = MIDIOut(0);
~dirt.soundLibrary.addMIDI(\midi, ~midiOut);

~midiOut.latency = 0.0;

~midiOut.control(1,2,64)


b.free;
b=Bus.control(s,4);


~noise={LFDNoise3.kr([0.1,0.1,0.1,0.1]).range(0, 127)}.play(s,b);

(
Task({ { ~midiOut.control(1, 1, b.getnSynchronous(4).at(0).round.asInteger); 0.1.wait;}.loop }).start;

Task({ { ~midiOut.control(1, 2, b.getnSynchronous(4).at(1).round.asInteger); 0.1.wait;}.loop }).start;

Task({ { ~midiOut.control(1, 3, b.getnSynchronous(4).at(2).round.asInteger); 0.1.wait;}.loop }).start;

Task({ { ~midiOut.control(1, 4, b.getnSynchronous(4).at(3).round.asInteger); 0.1.wait;}.loop }).start;
)
