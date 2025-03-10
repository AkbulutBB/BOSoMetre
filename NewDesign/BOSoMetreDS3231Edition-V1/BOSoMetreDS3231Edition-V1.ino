#include <SPI.h>
#include <SD.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <RTClib.h>

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
char patientID[20];  // Adjust size as needed

// Function Prototypes
void setupSensorPins();
void readColor(int &colorCount, int S2State, int S3State);
void checkShutdown();
void blinkLED(int delayTime);
void getTimestamp(char* buffer, size_t bufferSize);
void displayDataOnLCD(int redCount, int greenCount, int blueCount, int clearCount, int turbidityPercent);
void readPatientID();

void setup() {
  // Initialize serial communication
  Serial.begin(9600);
  // Give time for the serial monitor to open
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

  // Prompt for patient ID
  readPatientID();
}

void loop() {
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

    // Build the data string with timestamp and patient ID
    snprintf(dataString, sizeof(dataString), "%s,%s,%d,%d,%d,%d,%d%%,%d%%,%d%%,%d%%",
             patientID, timestamp,
             redPeriodCount, greenPeriodCount, bluePeriodCount, clearPeriodCount,
             redChangePercent, greenChangePercent, blueChangePercent, turbidityPercent);

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

    // Display data on the LCD
    displayDataOnLCD(redPeriodCount, greenPeriodCount, bluePeriodCount, clearPeriodCount, turbidityPercent);

    // Display TXT-OK or TXT-ERR on the LCD
    lcd.clear();
    if (sdWriteSuccess) {
      lcd.setCursor(0, 0);
      lcd.print("TXT-OK");
    } else {
      lcd.setCursor(0, 0);
      lcd.print("TXT-ERR");
    }
    delay(1000); // Display the message for 1 second

    // Return to displaying data
    displayDataOnLCD(redPeriodCount, greenPeriodCount, bluePeriodCount, clearPeriodCount, turbidityPercent);

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

void displayDataOnLCD(int redCount, int greenCount, int blueCount, int clearCount, int turbidityPercent) {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ID:");
  lcd.print(patientID);

  lcd.setCursor(0, 1);
  lcd.print("Turb:");
  lcd.print(turbidityPercent);
  lcd.print("%");

  lcd.setCursor(10, 1);
  lcd.print("Clr:");
  lcd.print(clearCount);

  lcd.setCursor(0, 2);
  lcd.print("R:");
  lcd.print(redCount);
  lcd.print(" G:");
  lcd.print(greenCount);

  lcd.setCursor(0, 3);
  lcd.print("B:");
  lcd.print(blueCount);
  lcd.print(" T:");
  char timeBuffer[6];
  DateTime now = rtc.now();
  snprintf(timeBuffer, sizeof(timeBuffer), "%02d:%02d", now.hour(), now.minute());
  lcd.print(timeBuffer);
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
