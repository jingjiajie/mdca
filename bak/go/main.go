package main

import (
	"context"
	"os"
	"strings"
	"time"

	"github.com/kelindar/bitmap"
	"github.com/rocketlaunchr/dataframe-go/imports"
)

func main() {
	println("Reading CSV file...")
	startTime := time.Now()
	file, err := os.ReadFile("data/flights/flights.csv")
	if err != nil {
		println(err.Error())
		return
	}
	println("File opened successfully, cost: ", time.Now().Sub(startTime).String())
	startTime = time.Now()
	file_str := string(file)
	println("File convert string, cost: ", time.Now().Sub(startTime).String())
	startTime = time.Now()
	df, err := imports.LoadFromCSV(context.Background(), strings.NewReader(file_str))
	println("CSV file read successfully, cost: ", time.Now().Sub(startTime).String())
	println(df.Names())

	books := bitmap.Bitmap{}
	books.Set(0)
	books.Set(100)
	books.Set(500)

	max, _ := books.Max()
	list := make([]string, max+1)
	var i uint32
	for i = 0; i <= max; i++ {
		if books.Contains(i) {
			list[i] = "1"
		} else {
			list[i] = "0"
		}
	}
	println(strings.Join(list, ","))
}
