"use client"

import { useState, useRef } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

export default function HealthRiskPredictor() {
  const [file, setFile] = useState<File | null>(null)
  const [progress, setProgress] = useState(0)
  const [results, setResults] = useState<any>(null)
  const [activeTab, setActiveTab] = useState("upload")
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files) {
      setFile(event.target.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) return

    setActiveTab("training")
    // Simulating file upload and processing
    for (let i = 0; i <= 100; i += 10) {
      setProgress(i)
      await new Promise(resolve => setTimeout(resolve, 500))
    }

    // Simulating results
    setResults({
      accuracy: 0.85,
      featureImportance: [
        { name: 'mean radius', importance: 0.2 },
        { name: 'mean texture', importance: 0.15 },
        { name: 'mean perimeter', importance: 0.3 },
        { name: 'mean area', importance: 0.25 },
        { name: 'mean smoothness', importance: 0.1 },
      ]
    })

    setActiveTab("results")
  }

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <CardTitle>Health Risk Predictor</CardTitle>
        <CardDescription>Upload your health data CSV file for analysis</CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="upload">Upload</TabsTrigger>
            <TabsTrigger value="training">Training</TabsTrigger>
            <TabsTrigger value="results">Results</TabsTrigger>
          </TabsList>
          <TabsContent value="upload">
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Label htmlFor="picture">Health Data CSV</Label>
              <Input 
                id="picture" 
                type="file" 
                accept=".csv"
                ref={fileInputRef}
                onChange={handleFileChange}
              />
            </div>
          </TabsContent>
          <TabsContent value="training">
            <div className="flex flex-col items-center gap-4">
              <Progress value={progress} className="w-full" />
              <p>Training in progress: {progress}%</p>
            </div>
          </TabsContent>
          <TabsContent value="results">
            {results && (
              <div className="space-y-4">
                <div>
                  <h3 className="text-lg font-semibold">Model Accuracy</h3>
                  <p>{(results.accuracy * 100).toFixed(2)}%</p>
                </div>
                <div>
                  <h3 className="text-lg font-semibold">Feature Importance</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={results.featureImportance}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="importance" fill="#8884d8" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
      <CardFooter>
        <Button onClick={handleUpload} disabled={!file || activeTab !== "upload"}>
          Analyze Data
        </Button>
      </CardFooter>
    </Card>
  )
}
