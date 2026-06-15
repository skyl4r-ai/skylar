       IDENTIFICATION DIVISION.
       PROGRAM-ID. SUM-TO-N.

       ENVIRONMENT DIVISION.
       
       INPUT-OUTPUT SECTION.

       DATA DIVISION.

       LINKAGE SECTION.

       01 LINKED-ITEMS.
           05 L-N PIC S9(10).
           05 RESULT PIC S9(10).

      * sum_to_n is a function that sums numbers from 1 to n.
      * >>> sum_to_n(30)
      * 465
      * >>> sum_to_n(100)
      * 5050
      * >>> sum_to_n(5)
      * 15
      * >>> sum_to_n(10)
      * 55
      * >>> sum_to_n(1)
      * 1
      * 

      * Complete the WORKING-STORAGE SECTION and the PROCEDURE DIVISION
      * Store the result in the RESULT variable and mark the end of your program with END PROGRAM

       WORKING-STORAGE SECTION.
       
       