#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>
#include <EEPROM.h>
#include <avr/wdt.h>  // Watchdog timer library

// Define pins connected to the TCS3200 sensor
#define S0 5
#define S1 6
#define S2 7
#define S3 8
#define sensorOut 9
#define shutdownPin 10  // Define the shutdown pin
#define greenLEDPin 11  // Define the green LED pin

// Calibration values for clear CSF readings (TO-DO: Calibrate these)
int baselineClearPeriodCount = 220; // Initialize to prevent division by zero
int baselineRedPeriodCount = 68;   
int baselineGreenPeriodCount = 76; 
int baselineBluePeriodCount = 90;  

// Global variable to track the last retry time for SD initialization
unsigned long lastRetryTime = 0;
const unsigned long retryInterval = 60000; // 1 minute interval for retry

// Timeout setting for the pulseIn() function (microseconds)
const int pulseTimeout = 2000; 

// Specifies the chip select pin for the SD card module
const int chipSelect = 12;  

// Variables for interacting with the SD card
File sensorDataFile;
char dataString[150];  // Increased size to accommodate longer strings

// Initialize the LCD
LiquidCrystal_I2C lcd(0x27, 20, 4);  // Change the address if needed

// Initialize the RTC
RTC_DS3231 rtc;

// Variable to track shutdown state
bool isShutdown = false;

// Variable to store patient ID
const int patientIDAddress = 0;  // EEPROM address for patient ID
char patientID[20];  // Adjust size as needed

// Function Prototypes
void setupSensorPins();
void readColor(int &colorCount, int S2State, int S3State);
void checkShutdown();
void blinkLED(int delayTime);
void getTimestamp(char* buffer, size_t bufferSize);
void displayDataOnLCD(int redCount, int greenCount, int blueCount, int clearCount, int turbidityPercent);
void readPatientID();
void updatePatientID();

// Define the analog pin for voltage reading
const int analogPin = A0;  // Pin connected to the resistor and white LED junction
float voltage;
float current;
float resistorValue = 68000.0;  // 68k ohms

void setup() {
  // Clear the watchdog reset flag and disable the watchdog timer during setup
  MCUSR &= ~(1<<WDRF);  // Clear the watchdog reset flag
  wdt_disable();        // Disable the watchdog during initialization

  // Initialize serial communication
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect. Needed for native USB port only
  }
  Serial.println("Serial communication initialized.");

  // Set up the sensor pins
  setupSensorPins();
  Serial.println("Sensor pins setup completed.");

  // Set up the shutdown pin and green LED pin
  pinMode(shutdownPin, INPUT_PULLUP); // Using internal pull-up resistor
  pinMode(greenLEDPin, OUTPUT);

  // Turn on the green LED to indicate standby mode
  digitalWrite(greenLEDPin, HIGH);

  // Initialize SD card
  Serial.print("Initializing SD card...");
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card initialization failed or not present!");
    lcd.init();
    lcd.backlight();
    lcd.clear();
    lcd.print("SD card failed!");

    // Blink the green LED fast to indicate an error
    while (1) {
      blinkLED(200); // Fast blink
    }
  }
  Serial.println("Initialization done.");

  // Initialize the LCD
  lcd.init();
  lcd.backlight();
  lcd.clear();
  lcd.print("Initializing...");

  // Initialize RTC
  if (!rtc.begin()) {
    Serial.println("Couldn't find RTC");
    while (1); // Halt if RTC not found
  }

  if (rtc.lostPower()) {
    Serial.println("RTC lost power, setting the time!");
    rtc.adjust(DateTime(F(__DATE__), F(__TIME__)));
  }

  // Load patient ID from EEPROM or prompt for a new one
  EEPROM.get(patientIDAddress, patientID);
  if (patientID[0] == '\0') { // If patient ID is not set, prompt for entry
    readPatientID();
    EEPROM.put(patientIDAddress, patientID);
  } else {
    Serial.print("Patient ID loaded from EEPROM: ");
    Serial.println(patientID);
    lcd.clear();
    lcd.print("Patient ID: ");
    lcd.setCursor(0, 1);
    lcd.print(patientID);
  }

  // Enable the watchdog timer (e.g., with an 8-second timeout)
  wdt_enable(WDTO_8S);
}

void loop() {
  // Reset the watchdog timer to avoid resetting the microcontroller
  wdt_reset();

  // Check for commands from the serial terminal (for patient ID update)
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    if (command.equalsIgnoreCase("UPDATE")) {
      updatePatientID();
    }
  }

  // Check if shutdown switch is pressed
  checkShutdown();

  // If the shutdown switch is not pressed, run the main loop
  if (!isShutdown) {
    // Indicate normal operation with long blinks
    digitalWrite(greenLEDPin, HIGH);
    delay(1000);
    digitalWrite(greenLEDPin, LOW);
    delay(1000);

    // Variables to store counts for each color and clear reading
    int redPeriodCount = 0;
    int greenPeriodCount = 0;
    int bluePeriodCount = 0;
    int clearPeriodCount = 0;

    // Read Red, Green, Blue, and Clear colors
    readColor(redPeriodCount, LOW, LOW);
    readColor(greenPeriodCount, HIGH, HIGH);
    readColor(bluePeriodCount, LOW, HIGH);
    readColor(clearPeriodCount, HIGH, LOW);

    // Calculate and log data
    int turbidityPercent = (clearPeriodCount - baselineClearPeriodCount) * 100 / baselineClearPeriodCount;
    int redChangePercent = (redPeriodCount - baselineRedPeriodCount) * 100 / baselineRedPeriodCount;
    int greenChangePercent = (greenPeriodCount - baselineGreenPeriodCount) * 100 / baselineGreenPeriodCount;
    int blueChangePercent = (bluePeriodCount - baselineBluePeriodCount) * 100 / baselineBluePeriodCount;

    // Get the current timestamp
    char timestamp[20];
    getTimestamp(timestamp, sizeof(timestamp));

    // Read the voltage drop across the 68k ohm resistor
    int analogValue = analogRead(analogPin);
    voltage = analogValue * (5.0 / 1023.0);  // Convert analog value to voltage
    current = voltage / resistorValue;  // Calculate current using Ohm's Law (I = V/R)

    // Build the data string with timestamp, patient ID, and additional voltage/current data
    snprintf(dataString, sizeof(dataString), "%s,%s,%d,%d,%d,%d,%d%%,%d%%,%d%%,%d%%,%.2fV,%.2fmA",
             patientID, timestamp,
             redPeriodCount, greenPeriodCount, bluePeriodCount, clearPeriodCount,
             redChangePercent, greenChangePercent, blueChangePercent, turbidityPercent,
             voltage, current * 1000);  // Convert current to mA and format

    // Debug statement to print the data string
    Serial.print("Data String: ");
    Serial.println(dataString);

    // Reinitialize SD card if necessary
    if (!SD.begin(chipSelect)) {
      unsigned long currentMillis = millis();
      if (currentMillis - lastRetryTime >= retryInterval) {
        Serial.println("SD card reinitialization failed! Retrying in 1 minute...");
        lcd.setCursor(0, 3);
        lcd.print("TXT-ERR: SD INIT");
        lastRetryTime = currentMillis;  // Update the last retry time
      }
      delay(2000);  // Give some time to see the error
      return;  // Exit the loop if SD initialization failed
    }

    // Log data to SD card
    bool sdWriteSuccess = false; // Flag to track SD card write status
    sensorDataFile = SD.open("SENSOR.TXT", FILE_WRITE);
    if (sensorDataFile) {
      sensorDataFile.println(dataString);
      sensorDataFile.close();
      Serial.println("Text file write complete");
      sdWriteSuccess = true;
    } else {
      // if the file didn't open, print an error:
      Serial.println("Error opening text file.");
      sdWriteSuccess = false;
    }

    // Display data on the LCD with new layout
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("ID: ");
    char tempID[11];  // Temporary buffer to hold the first 10 characters + null terminator
    strncpy(tempID, patientID, 10);  // Copy the first 10 characters
    tempID[10] = '\0';  // Ensure null-terminated string
    lcd.print(tempID);  // Print the trimmed ID

    
    lcd.setCursor(0, 1);
    lcd.print("Clr:");
    lcd.print(clearPeriodCount);
    lcd.print(" Turb:");
    lcd.print(turbidityPercent);
    lcd.print("%");

    lcd.setCursor(0, 2);
    lcd.print("R:");
    lcd.print(redPeriodCount);
    lcd.print(" G:");
    lcd.print(greenPeriodCount);
    lcd.print(" B:");
    lcd.print(bluePeriodCount);

    lcd.setCursor(0, 3);
    lcd.print("V:");
    lcd.print(voltage, 2);  // Display the voltage with 2 decimal places
    lcd.print(" I:");
    lcd.print(current * 1000, 2);  // Display the current in mA
    lcd.print(" T:");
    
    char timeBuffer[6];
    DateTime now = rtc.now();
    snprintf(timeBuffer, sizeof(timeBuffer), "%02d:%02d", now.hour(), now.minute());
    lcd.print(timeBuffer);

    // Short delay before the next loop iteration
    delay(1000);
  } else {
    // If the shutdown switch is pressed, indicate standby mode
    digitalWrite(greenLEDPin, HIGH);
  }
}

void setupSensorPins() {
  pinMode(S0, OUTPUT);
  pinMode(S1, OUTPUT);
  pinMode(S2, OUTPUT);
  pinMode(S3, OUTPUT);
  pinMode(sensorOut, INPUT);
}

void readColor(int &colorCount, int S2State, int S3State) {
  // Setting frequency scaling to 20%
  digitalWrite(S0, HIGH);
  digitalWrite(S1, LOW);

  digitalWrite(S2, S2State);
  digitalWrite(S3, S3State);
  unsigned long startTime = millis();
  colorCount = 0;
  int dynamicMeasurementTime = 50;
  unsigned long currentTime = millis();

  while (currentTime - startTime < dynamicMeasurementTime) {
    int pulseWidth = pulseIn(sensorOut, LOW, pulseTimeout);
    if (pulseWidth > 0) {
      colorCount++;
    }
    currentTime = millis();
  }
}

void checkShutdown() {
  if (digitalRead(shutdownPin) == HIGH) {
    // Shutdown switch pressed
    if (!isShutdown) {
      Serial.println("Shutdown switch pressed. Stopping SD card operations.");
      lcd.clear();
      lcd.print("On standby...");

      // Close the file if it's open
      if (sensorDataFile) {
        sensorDataFile.close();
      }

      // Turn off the green LED to indicate shutdown
      digitalWrite(greenLEDPin, LOW);

      isShutdown = true;
    }
  } else {
    // Shutdown switch released
    if (isShutdown) {
      Serial.println("Shutdown switch released. Restarting operations.");
      lcd.clear();
      lcd.print("Restarting...");

      // Turn on the green LED to indicate normal operation
      digitalWrite(greenLEDPin, HIGH);

      isShutdown = false;
    }
  }
}

void blinkLED(int delayTime) {
  digitalWrite(greenLEDPin, HIGH);
  delay(delayTime);
  digitalWrite(greenLEDPin, LOW);
  delay(delayTime);
}

void getTimestamp(char* buffer, size_t bufferSize) {
  DateTime now = rtc.now();
  snprintf(buffer, bufferSize, "%04d/%02d/%02d %02d:%02d:%02d",
           now.year(), now.month(), now.day(),
           now.hour(), now.minute(), now.second());
}

void readPatientID() {
  Serial.println("Please enter patient ID and press ENTER:");
  while (Serial.available() == 0) {
    // Wait for input
  }
  String tempID = Serial.readStringUntil('\n');
  tempID.trim();
  tempID.toCharArray(patientID, sizeof(patientID));
  Serial.print("Patient ID set to: ");
  Serial.println(patientID);
  lcd.clear();
  lcd.print("Patient ID: ");
  lcd.setCursor(0, 1);
  lcd.print(patientID);
}

void updatePatientID() {
  Serial.println("Updating patient ID...");
  readPatientID(); // Prompt for new patient ID
  EEPROM.put(patientIDAddress, patientID); // Save updated patient ID to EEPROM
}
