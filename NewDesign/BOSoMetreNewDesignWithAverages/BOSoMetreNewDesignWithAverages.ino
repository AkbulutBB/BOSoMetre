#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <EEPROM.h>
#include <avr/wdt.h>

// Pin definitions
#define S0 5
#define S1 6
#define S2 7
#define S3 8
#define sensorOut 9
#define shutdownPin 10
#define greenLEDPin 11
#define analogPin A5

// EEPROM addresses
#define BASELINE_RED_ADDR 20
#define BASELINE_GREEN_ADDR 24
#define BASELINE_BLUE_ADDR 28
#define BASELINE_CLEAR_ADDR 32

// Timing and sampling
const unsigned long RECORD_INTERVAL = 60000; // 60,000 ms = 1 minute
const unsigned long SAMPLE_INTERVAL = 2000;  // Sample every 2 seconds
const int SAMPLES_PER_MINUTE = 30;          // 30 samples per minute

// Baseline values (will be loaded from EEPROM)
int baselineClearPeriodCount;
int baselineRedPeriodCount;   
int baselineGreenPeriodCount; 
int baselineBluePeriodCount;  

// Global variables
unsigned long lastRecordTime = 0;
unsigned long lastRetryTime = 0;
const int pulseTimeout = 2000; 
const int chipSelect = 12;  

File sensorDataFile;
char dataString[90];
LiquidCrystal_I2C lcd(0x27, 20, 4);
RTC_DS3231 rtc;
bool isShutdown = false;
const int patientIDAddress = 0;
char patientID[10];

// Averaging arrays
int redReadings[SAMPLES_PER_MINUTE];
int greenReadings[SAMPLES_PER_MINUTE];
int blueReadings[SAMPLES_PER_MINUTE];
int clearReadings[SAMPLES_PER_MINUTE];
float voltageReadings[SAMPLES_PER_MINUTE];
int currentSample = 0;

void setup() {
  MCUSR &= ~(1<<WDRF);
  wdt_disable();

  Serial.begin(9600);
  
  setupSensorPins();
  
  pinMode(shutdownPin, INPUT_PULLUP);
  pinMode(greenLEDPin, OUTPUT);
  digitalWrite(greenLEDPin, HIGH);

  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("Starting.");

  if (!rtc.begin()) {
    lcd.clear();
    lcd.print("RTC-ERR");
    while (1);
  }

  if (rtc.lostPower()) {
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  if (!SD.begin(chipSelect)) {
    lcd.clear();
    lcd.print("SD-ERR");
    while (1) {
      blinkLED(200);
    }
  }

  EEPROM.get(patientIDAddress, patientID);
  if (patientID[0] == '\0') {
    readPatientID();
    EEPROM.put(patientIDAddress, patientID);
  }

  readBaselinesFromEEPROM();
  clearArrays();
  lastRecordTime = millis();

  wdt_enable(WDTO_8S);
}

void loop() {
  wdt_reset();

  // Simplified command handling without String objects
  if (Serial.available() > 0) {
    char cmd = Serial.read();  // Read the first command character
    
    // Process the command based on the initial character
    if (cmd == 'U') {
      updatePatientID();
    }
    else if (cmd == 'B') {
      showBaselines();
    }
    else if (cmd == 'S') {
      char type = Serial.read();  // Expecting 'R', 'G', 'B', or 'C' after 'S'
      
      // Wait until we find a space to skip to the value part
      while (Serial.read() != ' ') {}

      // Parse integer value directly from serial input
      int value = 0;
      while (Serial.available()) {
        char c = Serial.read();
        if (c >= '0' && c <= '9') {
          value = value * 10 + (c - '0');
        }
      }

      // Update the baseline if the value is valid
      if (value > 0) {
        switch (type) {
          case 'R': 
            EEPROM.put(BASELINE_RED_ADDR, value);
            baselineRedPeriodCount = value;
            break;
          case 'G': 
            EEPROM.put(BASELINE_GREEN_ADDR, value);
            baselineGreenPeriodCount = value;
            break;
          case 'B': 
            EEPROM.put(BASELINE_BLUE_ADDR, value);
            baselineBluePeriodCount = value;
            break;
          case 'C': 
            EEPROM.put(BASELINE_CLEAR_ADDR, value);
            baselineClearPeriodCount = value;
            break;
          default:
            Serial.println("Invalid type. Use R, G, B, or C.");
            return;
        }
        // Display updated baselines after modification
        showBaselines();
      } else {
        Serial.println("Invalid value.");
      }
    }
  }

  checkShutdown();

  if (!isShutdown) {
    unsigned long currentTime = millis();
    
    if (currentTime < lastRecordTime) {
      lastRecordTime = currentTime;
    }

    if (currentTime - lastRecordTime >= SAMPLE_INTERVAL) {
      digitalWrite(greenLEDPin, HIGH);
      
      float v = analogRead(analogPin) * (5.0 / 1023.0);
      
      if (v >= 2.5) {
        int r = 0, g = 0, b = 0, c = 0;
        readColor(r, LOW, LOW);
        readColor(g, HIGH, HIGH);
        readColor(b, LOW, HIGH);
        readColor(c, HIGH, LOW);

        if (currentSample < SAMPLES_PER_MINUTE) {
          redReadings[currentSample] = r;
          greenReadings[currentSample] = g;
          blueReadings[currentSample] = b;
          clearReadings[currentSample] = c;
          voltageReadings[currentSample] = v;
          currentSample++;
          
          lcd.setCursor(0,3);
          lcd.print("P:"); 
          lcd.print((currentSample * 100) / SAMPLES_PER_MINUTE);
          lcd.print("%  ");
        }
      } else {
        lcd.clear();
        lcd.print("Low voltage!");
        lcd.setCursor(0,1);
        lcd.print(v);
        lcd.print("V");
        delay(1000);
      }
      
      digitalWrite(greenLEDPin, LOW);
      lastRecordTime = currentTime;
    }

    if (currentSample >= SAMPLES_PER_MINUTE) {
      float avgRed = calculateAverage(redReadings, SAMPLES_PER_MINUTE);
      float avgGreen = calculateAverage(greenReadings, SAMPLES_PER_MINUTE);
      float avgBlue = calculateAverage(blueReadings, SAMPLES_PER_MINUTE);
      float avgClear = calculateAverage(clearReadings, SAMPLES_PER_MINUTE);
      float avgVoltage = calculateAverage(voltageReadings, SAMPLES_PER_MINUTE);

      int turbidityPercent = (avgClear - baselineClearPeriodCount) * 100 / baselineClearPeriodCount;
      int redChangePercent = (avgRed - baselineRedPeriodCount) * 100 / baselineRedPeriodCount;
      int greenChangePercent = (avgGreen - baselineGreenPeriodCount) * 100 / baselineGreenPeriodCount;
      int blueChangePercent = (avgBlue - baselineBluePeriodCount) * 100 / baselineBluePeriodCount;

      char timestamp[20];
      getTimestamp(timestamp, sizeof(timestamp));

      char voltageString[8];
      dtostrf(avgVoltage, 4, 2, voltageString);

      snprintf(dataString, sizeof(dataString), "%s,%s,%.1f,%.1f,%.1f,%.1f,%d,%d,%d,%d,%s",
               patientID, timestamp,
               avgRed, avgGreen, avgBlue, avgClear,
               redChangePercent, greenChangePercent, blueChangePercent, turbidityPercent,
               voltageString);

      if (SD.begin(chipSelect)) {
        sensorDataFile = SD.open("SENSOR.TXT", FILE_WRITE);
        if (sensorDataFile) {
          sensorDataFile.println(dataString);
          sensorDataFile.close();
          Serial.println("TXT-OK");
        }
      }

      updateLCDDisplay(avgRed, avgGreen, avgBlue, avgClear, turbidityPercent, voltageString);
      clearArrays();
    }
    
    delay(50);
  } else {
    digitalWrite(greenLEDPin, HIGH);
  }
}

void setupSensorPins() {
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(sensorOut, INPUT);
  
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);
}

void readColor(int &colorCount, int S2State, int S3State) {
  digitalWrite(S2, S2State);
  digitalWrite(S3, S3State);
  
  colorCount = 0;
  unsigned long endTime = millis() + 50;
  
  while (millis() < endTime) {
    if (pulseIn(sensorOut, LOW, pulseTimeout) > 0) {
      colorCount++;
    }
  }
}

void checkShutdown() {
  if (digitalRead(shutdownPin) == HIGH) {
    if (!isShutdown) {
      lcd.clear();
      lcd.print("Standby");
      if (sensorDataFile) {
        sensorDataFile.close();
      }
      digitalWrite(greenLEDPin, LOW);
      isShutdown = true;
    }
  } else {
    if (isShutdown) {
      lcd.clear();
      lcd.print("Restarting");
      digitalWrite(greenLEDPin, HIGH);
      isShutdown = false;
      delay(2000);
      lcd.clear();
    }
  }
}

void clearArrays() {
  for(int i = 0; i < SAMPLES_PER_MINUTE; i++) {
    redReadings[i] = 0;
    greenReadings[i] = 0;
    blueReadings[i] = 0;
    clearReadings[i] = 0;
    voltageReadings[i] = 0;
  }
  currentSample = 0;
}

float calculateAverage(int readings[], int arraySize) {
  long sum = 0;
  for(int i = 0; i < arraySize; i++) {
    sum += readings[i];
  }
  return (float)sum / arraySize;
}

float calculateAverage(float readings[], int arraySize) {
  float sum = 0;
  for(int i = 0; i < arraySize; i++) {
    sum += readings[i];
  }
  return sum / arraySize;
}

void blinkLED(int delayTime) {
  digitalWrite(greenLEDPin, HIGH);
  delay(delayTime);
  digitalWrite(greenLEDPin, LOW);
  delay(delayTime);
}

void getTimestamp(char* buffer, size_t bufferSize) {
  DateTime now = rtc.now();
  snprintf(buffer, bufferSize, "%02d%02d%02d_%02d%02d",
           now.year() % 100, now.month(), now.day(),
           now.hour(), now.minute());
}

void readPatientID() {
  Serial.println("Enter ID:");
  
  while(Serial.available()) Serial.read();
  
  while (Serial.available() == 0) {
    wdt_reset();
  }
  
  String tempID = Serial.readStringUntil('\n');
  tempID.trim();
  
  if(tempID.length() > 10) {
    tempID = tempID.substring(0, 10);
  }
  
  tempID.toCharArray(patientID, sizeof(patientID));
  
  Serial.print("ID: ");
  Serial.println(patientID);
  
  lcd.clear();
  lcd.print("ID Set: ");
  lcd.setCursor(0, 1);
  lcd.print(patientID);
  delay(2000);
}

void updatePatientID() {
  readPatientID();
  EEPROM.put(patientIDAddress, patientID);
}

void readBaselinesFromEEPROM() {
  EEPROM.get(BASELINE_RED_ADDR, baselineRedPeriodCount);
  EEPROM.get(BASELINE_GREEN_ADDR, baselineGreenPeriodCount);
  EEPROM.get(BASELINE_BLUE_ADDR, baselineBluePeriodCount);
  EEPROM.get(BASELINE_CLEAR_ADDR, baselineClearPeriodCount);
  
  if (baselineRedPeriodCount <= 0 || baselineRedPeriodCount == 255) baselineRedPeriodCount = 68;
  if (baselineGreenPeriodCount <= 0 || baselineGreenPeriodCount == 255) baselineGreenPeriodCount = 76;
  if (baselineBluePeriodCount <= 0 || baselineBluePeriodCount == 255) baselineBluePeriodCount = 90;
  if (baselineClearPeriodCount <= 0 || baselineClearPeriodCount == 255) baselineClearPeriodCount = 220;
}

void showBaselines() {
  Serial.print("R:"); Serial.println(baselineRedPeriodCount);
  Serial.print("G:"); Serial.println(baselineGreenPeriodCount);
  Serial.print("B:"); Serial.println(baselineBluePeriodCount);
  Serial.print("C:"); Serial.println(baselineClearPeriodCount);
}

void updateLCDDisplay(float redCount, float greenCount, float blueCount, 
                     float clearCount, int turbidity, char* voltageStr) {
    lcd.clear();
    lcd.print("ID:");
    lcd.print(patientID);
    
    lcd.setCursor(0, 1);
    lcd.print("C:");
    lcd.print(clearCount, 0);  
    lcd.print(" T:");          
    lcd.print(turbidity);
    lcd.print("%");

    lcd.setCursor(0, 2);
    lcd.print("R: ");         
    lcd.print(redCount, 0);
    lcd.print("G: ");
    lcd.print(greenCount, 0);
    lcd.print("B: ");
    lcd.print(blueCount, 0);

    lcd.setCursor(0, 3);
    lcd.print("P:"); 
    lcd.print("100% ");

    lcd.setCursor(7, 3);
    lcd.print("V:");
    lcd.print(voltageStr);
    lcd.print(" ");
    
    lcd.print("T:");
    DateTime now = rtc.now();
    char timeStr[6];
    sprintf(timeStr, "%02d%02d", now.hour(), now.minute());
    lcd.print(timeStr);
}