import os
import rotaryio
import board
import time
import audiobusio
from digitalio import DigitalInOut, Direction, Pull
from adafruit_debouncer import Debouncer
from audiobusio import PDMIn
from array import array
from math import log, sqrt, fabs
from time import monotonic
from supervisor import reload
import board
from rainbowio import colorwheel
from ulab import numpy as np
#  if using CP8 and above:
from ulab.utils import spectrogram
# neopixels imports
import neopixel
from adafruit_pixel_framebuf import PixelFramebuffer
from adafruit_led_animation.color import (
    BLACK,
    RED,
    ORANGE,
    BLUE,
    PURPLE,
    WHITE,
    PINK,
    OLD_LACE,
    CYAN,
    MAGENTA,
    GREEN,
    TEAL,
    YELLOW,
    GOLD
)
from adafruit_led_animation.animation.rainbowsparkle import RainbowSparkle
from adafruit_led_animation.animation.comet import Comet

import adafruit_pcf8523
import busio
import audiomp3
import audiomixer
import audiocore
from adafruit_ticks import ticks_ms, ticks_add, ticks_diff
import adafruit_lis3dh


i2c = busio.I2C(board.SCL, board.SDA)
rtc = adafruit_pcf8523.PCF8523(i2c)
# accolerometer
int1 = DigitalInOut(board.ACCELEROMETER_INTERRUPT)
lis3dh = adafruit_lis3dh.LIS3DH_I2C(i2c, int1=int1)
lis3dh.range = adafruit_lis3dh.RANGE_2_G

# rotary encoder
encoder = rotaryio.IncrementalEncoder(board.D10, board.D11)
last_knob_position = 0

days = ("Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
if False:   # change to True if you want to write the time!
    #                     year, mon, date, hour, min, sec, wday, yday, isdst
    t = time.struct_time((2023,  8,   17,   16,  31,  0,    0,   -1,    -1))
    # you must set year, mon, date, hour, min, sec and weekday
    # yearday is not supported, isdst can be set but we don't do anything with it at this time

    print("Setting time to:", t)     # uncomment for debugging
    rtc.datetime = t
    print()

# encoder button
knob_button= DigitalInOut(board.EXTERNAL_BUTTON)
knob_button.direction = Direction.INPUT
knob_button.pull = Pull.UP
button = Debouncer(knob_button)

#  mic setup
mic = PDMIn(board.TX, board.A1, sample_rate=16000, bit_depth=16)

# FFT/SPECTRUM CONFIG ----

fft_size = 256  # Sample size for Fourier transform, MUST be power of two
spectrum_size = fft_size // 2  # Output spectrum is 1/2 of FFT result
# Bottom of spectrum tends to be noisy, while top often exceeds musical
# range and is just harmonics, so clip both ends off:
low_bin = 20  # Lowest bin of spectrum that contributes to graph
high_bin = 75  # Highest bin "


# HARDWARE SETUP ---------
rec_buf = array("H", [0] * fft_size)  # 16-bit audio samples

#  enable power to neopixels
external_power = DigitalInOut(board.EXTERNAL_POWER)
external_power.direction = Direction.OUTPUT
external_power.value = True
#  neopixel setup
pixel_pin = board.EXTERNAL_NEOPIXELS
grid_width = 8
grid_height = 8
pixel_num = grid_width * grid_height
pixels = neopixel.NeoPixel(pixel_pin, pixel_num, brightness=0.5, auto_write=False, pixel_order=neopixel.GRB)
pixel_framebuf = PixelFramebuffer(
    pixels,
    grid_width,
    grid_height,
    reverse_x=False,
    alternating=False,
    rotation=1
)
animator =  RainbowSparkle(pixels, speed=0.1, num_sparkles=15)
alarm_animation = Comet(pixels, speed=0.1, color=GOLD, tail_length=7, bounce=True)

pixels.fill(BLACK)
pixels.show()

#  function to average mic levels
def mean(values):
    return sum(values) / len(values)

#  function to return mic level
def normalized_rms(values):
    minbuf = int(mean(values))
    samples_sum = sum(
        float(sample - minbuf) * (sample - minbuf)
        for sample in values
    )
    return sqrt(samples_sum / len(values))

# FFT/SPECTRUM SETUP -----

# To keep the display lively, tables are precomputed where each column of
# the matrix (of which there are few) is the sum value and weighting of
# several bins from the FFT spectrum output (of which there are many).
# The tables also help visually linearize the output so octaves are evenly
# spaced, as on a piano keyboard, whereas the source spectrum data is
# spaced by frequency in Hz.
column_table = []

spectrum_bits = log(spectrum_size, 2)  # e.g. 7 for 128-bin spectrum
# Scale low_bin and high_bin to 0.0 to 1.0 equivalent range in spectrum
low_frac = log(low_bin, 2) / spectrum_bits
frac_range = log(high_bin, 2) / spectrum_bits - low_frac

# color for each column
color_array = [ORANGE, YELLOW, GOLD, PINK, PURPLE, MAGENTA, CYAN, TEAL]

for column in range(grid_width):
    # Determine the lower and upper frequency range for this column, as
    # fractions within the scaled 0.0 to 1.0 spectrum range. 0.95 below
    # creates slight frequency overlap between columns, looks nicer.
    lower = low_frac + frac_range * (column / grid_width * 0.95)
    upper = low_frac + frac_range * ((column + 1) / grid_width)
    mid = (lower + upper) * 0.5  # Center of lower-to-upper range
    half_width = (upper - lower) * 0.5  # 1/2 of lower-to-upper range
    # Map fractions back to spectrum bin indices that contribute to column
    first_bin = int(2 ** (spectrum_bits * lower) + 1e-4)
    last_bin = int(2 ** (spectrum_bits * upper) + 1e-4)
    bin_weights = []  # Each spectrum bin's weighting will be added here
    for bin_index in range(first_bin, last_bin + 1):
        # Find distance from column's overall center to individual bin's
        # center, expressed as 0.0 (bin at center) to 1.0 (bin at limit of
        # lower-to-upper range).
        bin_center = log(bin_index + 0.5, 2) / spectrum_bits
        dist = abs(bin_center - mid) / half_width
        if dist < 1.0:  # Filter out a few math stragglers at either end
            # Bin weights have a cubic falloff curve within range:
            dist = 1.0 - dist  # Invert dist so 1.0 is at center
            bin_weights.append(((3.0 - (dist * 2.0)) * dist) * dist)
    # Scale bin weights so total is 1.0 for each column, but then mute
    # lower columns slightly and boost higher columns. It graphs better.
    total = sum(bin_weights)
    bin_weights = [
        (weight / total) * (0.8 + idx / grid_width * 1.4)
        for idx, weight in enumerate(bin_weights)
    ]
    # List w/five elements is stored for each column:
    # 0: Index of the first spectrum bin that impacts this column.
    # 1: A list of bin weights, starting from index above, length varies.
    # 2: Color for drawing this column on the LED matrix. The 225 is on
    #    purpose, providing hues from red to purple, leaving out magenta.
    # 3: Current height of the 'falling dot', updated each frame
    # 4: Current velocity of the 'falling dot', updated each frame
    column_table.append(
        [
            first_bin - low_bin,
            bin_weights,
            color_array[column],
            grid_height,
            0.0,
        ]
    )
# print(column_table)


# MAIN LOOP -------------

dynamic_level = 10  # For responding to changing volume levels
frames, start_time = 0, monotonic()  # For frames-per-second calc
magnitude_threshold = 50 # lower for more sensitivity to ambient should, higher for less

VISUALIZER = 'VISUALIZER'
CLOCK = 'CLOCK'
display_modes = [CLOCK, VISUALIZER]
current_mode_idx = 0

def next_mode(index):
    index += 1
    if (index >= len(display_modes)):
        index = 0
    return index

cleared = False

def _rgb_to_int(rgb):
        return rgb[0] << 16 | rgb[1] << 8 | rgb[2]

scroll_x_pos = -grid_width
word_timer = ticks_ms()
word_interval = 150

def scroll_word(word, color):
        global scroll_x_pos
        global word_timer
        global word_timer
        #print('scroll_word?')
        time_past = ticks_diff(ticks_ms(), word_timer)
        #print(time_past)
        if time_past >= word_interval:
            #print('incrementing?')
            total_scroll_len = (len(word) * 5) + len(word)

            if scroll_x_pos >= total_scroll_len:
                scroll_x_pos = -grid_width

            __scroll_framebuf(word, scroll_x_pos, 0, color)
            scroll_x_pos = scroll_x_pos + 1
            word_timer = ticks_add(word_timer, word_interval)

def __scroll_framebuf(word, shift_x, shift_y, color):
        pixel_framebuf.fill(0)
        color_int = _rgb_to_int(color)
        # negate x so that the word can be shown from left to right
        pixel_framebuf.text(word, -shift_x, shift_y, color_int)
        pixel_framebuf.display()

def clear():
    global cleared
    pixel_framebuf.fill(0x000000)
    pixel_framebuf.display()
    cleared = True

#audio stuff!
alarm = False
# 6:10pm alarm
alarm_minute = 5
alarm_hour = 18

wavs = []
for filename in os.listdir('/alarms'):
    if filename.lower().endswith('.wav') and not filename.startswith('.'):
        wavs.append("/alarms/"+filename)

audio = audiobusio.I2SOut(board.I2S_BIT_CLOCK, board.I2S_WORD_SELECT, board.I2S_DATA)
mixer = audiomixer.Mixer(voice_count=1, sample_rate=22050, channel_count=1,
                         bits_per_sample=16, samples_signed=True, buffer_size=32768)

mixer.voice[0].level = 1
track_number = 0
wav_filename = wavs[track_number]
wav_file = open(wav_filename, "rb")
wave = audiocore.WaveFile(wav_file)
audio.play(mixer)

def open_audio(num):
    n = wavs[num]
    f = open(n, "rb")
    w = audiocore.WaveFile(f)
    return w

stop_alarm = False
clock_position = ['UPRIGHT', 'SLEEP', 'SIDE']
current_clock_position = clock_position[0]
clock = ticks_ms()
prop_time = 1000
rotated = False

seconds_timer = ticks_ms()
minute_timer = ticks_ms()
seconds_interval = 1000
minute_interval = seconds_interval * 60
timer_counter = 0
drip_start = False
current_idx = 0
ticktock = False
timer_done = False
last_clock_position = current_clock_position
while True:
    button.update()
    t = rtc.datetime
    position = encoder.position

    if ticks_diff(ticks_ms(), clock) >= prop_time:
        x, y, z = [value / adafruit_lis3dh.STANDARD_GRAVITY for value in lis3dh.acceleration]
        # sleep!
        if z > 0.9:
            current_clock_position = 'SLEEP'
            #print(current_clock_position)
            external_power.value = False
        else:
            clock_sleep = False
            external_power.value = True

        # upright
        if fabs(x) > 0.9:
            current_clock_position = 'UPRIGHT'
            #print(current_clock_position)

        # side
        if fabs(y) > 0.9:
            current_clock_position = 'SIDE'
            #print(current_clock_position)
            pixel_framebuf.rotation = 0
        else:
            pixel_framebuf.rotation = 1

        if last_clock_position != current_clock_position:
            cleared = False
            clear()
            word_timer = ticks_ms()
            seconds_timer = ticks_ms()
            minute_timer = ticks_ms()

        last_clock_position = current_clock_position
        clock = ticks_add(clock, prop_time)

    if alarm_minute == t.tm_min and alarm_hour == t.tm_hour and alarm:
        alarm_animation.animate()
        word_timer = ticks_ms()
        seconds_timer = ticks_ms()
        minute_timer = ticks_ms()
        if(button.fell):
            stop_alarm = True

        continue

    if current_clock_position == 'SLEEP':
        if not cleared:
            clear()
            print('sleep!')
    elif current_clock_position == 'UPRIGHT':
        if button.fell:
            print('fell!')
            current_mode_idx = next_mode(current_mode_idx)
            cleared = False

        if (t.tm_hour == alarm_hour and t.tm_min == alarm_minute and alarm == False and stop_alarm == False):
            alarm = True
            mixer.voice[0].play(wave)
            pixel_framebuf.pixel(0,0,PINK)
            pixel_framebuf.display();
            print('alarm once!')

        if (display_modes[current_mode_idx] == VISUALIZER):
            word_timer = ticks_ms()

            if position != last_knob_position:
                delta = last_knob_position - position
                print('delta', delta)
                if delta < 1: # to the right
                    magnitude_threshold += 1
                else:
                    magnitude_threshold -= 1
                print('magnitude_threshold', magnitude_threshold)
                last_knob_position = position

            mic.record(rec_buf, fft_size)  # Record batch of 16-bit samples
            # used as a threshold for graphing
            magnitude = normalized_rms(rec_buf)
            #uncomment to see magnitude plotted
            # print((magnitude,))
            samples = np.array(rec_buf)  # Convert to ndarray
            # Compute spectrogram and trim results. Only the left half is
            # normally needed (right half is mirrored), but we trim further as
            # only the low_bin to high_bin elements are interesting to graph.
            spectrum = spectrogram(samples)[low_bin : high_bin + 1]
            # Linearize spectrum output. spectrogram() is always nonnegative,
            # but add a tiny value to change any zeros to nonzero numbers
            # (avoids rare 'inf' error)
            spectrum = np.log(spectrum + 1e-7)
            # Determine minimum & maximum across all spectrum bins, with limits
            lower = max(np.min(spectrum), 4)
            upper = min(max(np.max(spectrum), lower + 6), 20)

            # Adjust dynamic level to current spectrum output, keeps the graph
            # 'lively' as ambient volume changes. Sparkle but don't saturate.
            if upper > dynamic_level:
                # Got louder. Move level up quickly but allow initial "bump."
                dynamic_level = upper * 0.7 + dynamic_level * 0.3
            else:
                # Got quieter. Ease level down, else too many bumps.
                dynamic_level = dynamic_level * 0.6 + lower * 0.6

            # Apply vertical scale to spectrum data. Results may exceed
            # matrix height...that's OK, adds impact!
            #data = (spectrum - lower) * (7 / (dynamic_level - lower))
            data = (spectrum - lower) * ((grid_height + 2) / (dynamic_level - lower))
            #print(data)
            for column, element in enumerate(column_table):
                # Start BELOW matrix and accumulate bin weights UP, saves math
                first_bin = element[0]
                column_top = grid_height + 1
                for bin_offset, weight in enumerate(element[1]):
                    column_top -= data[first_bin + bin_offset] * weight

                if column_top < element[3]:  #       Above current falling dot?
                    element[3] = column_top - 0.5  # Move dot up
                    element[4] = 0  #                and clear out velocity
                else:
                    element[3] += element[4]  #      Move dot down
                    element[4] += 0.2  #             and accelerate

                column_top = int(column_top)  #      Quantize to pixel space


                for row in range(column_top):  #     Erase area above column
                    pixel_framebuf.pixel(column, row, 0)

                    if (magnitude > magnitude_threshold):
                        for row in range(column_top, grid_height):  #  Draw column
                            pixel_framebuf.pixel(column, row, element[2])
                    else:
                        pixel_framebuf.pixel(column, grid_height, element[2])
                #pixel_framebuf.pixel(column, int(element[3]), WHITE)  # Draw peak dot


            pixel_framebuf.display()  # Buffered mode MUST use show() to refresh matrix

            frames += 1
            #print(frames / (monotonic() - start_time), "FPS")

        elif (display_modes[current_mode_idx] == CLOCK):
            #clear()
            #print("The date is %s %d/%d/%d" % (days[t.tm_wday], t.tm_mday, t.tm_mon, t.tm_year))
            #print("The time is %d:%02d:%02d" % (t.tm_hour, t.tm_min, t.tm_sec))
            hour = t.tm_hour
            ampm = 'am'
            if hour > 12:
                hour = t.tm_hour - 12
            if t.tm_hour >= 12:
                ampm = 'pm'
            minute = t.tm_min
            if minute < 10:
                minute = f"0{t.tm_min}"
            timeword = f"{hour}:{minute}{ampm}"
            # scroll time
            scroll_word(timeword, GOLD)
    elif current_clock_position == 'SIDE':
        timer_color = MAGENTA
        if position != last_knob_position:
            delta = last_knob_position - position
            print('delta', delta)
            if delta < 1: # to the right
                timer_color = MAGENTA
                pixels[current_idx] = timer_color
                pixels.show()
                # don't go past grid_height * grid_width
                current_idx +=1
                if (current_idx >= grid_height * grid_width):
                    current_idx = (grid_height * grid_width) - 1
            else:
                timer_color = BLACK
                pixels[current_idx] = timer_color
                pixels.show()
                # don't go past 0
                current_idx -=1
                if(current_idx < 0):
                    current_idx = 0

            print('current_idx', current_idx)
            last_knob_position = position

        if button.fell:
            drip_start = not drip_start

            if timer_done:
                pixels.fill(0)
                pixels.show()
                timer_done = False

            if not drip_start:
                # restart
                pixels.fill(0)
                pixels.show()
                current_idx = 0
            print('current_idx', current_idx)
            print('drip_start', drip_start)


        if drip_start:

            if ticks_diff(ticks_ms(), seconds_timer) >= seconds_interval:
                ticktock = not ticktock
                seconds_timer = ticks_add(seconds_timer, seconds_interval)

            if ticktock:
                pixels[current_idx] = MAGENTA
                pixels.show()
            else:
                pixels[current_idx] = BLACK
                pixels.show()

            if ticks_diff(ticks_ms(), minute_timer) >= minute_interval:
                print('minute!')
                print(current_idx)
                minute_timer = ticks_add(minute_timer, minute_interval)
                pixels[current_idx] = BLACK
                pixels.show()
                current_idx -= 1
                if current_idx <= 0: # finished!
                    drip_start = False
                    timer_done = True

        if timer_done:
            animator.animate()